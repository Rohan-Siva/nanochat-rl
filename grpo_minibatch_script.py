import os
import itertools
import torch
import wandb
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

def main():
    if os.path.basename(os.getcwd()) == "scripts":
        os.chdir("..")

    print(f"Current working directory: {os.getcwd()}")

    # configs
    run_name = "grpo_gsm8k_minibatch" 
    source = "sft" 
    dtype = "bfloat16"
    device_batch_size = 4 
    examples_per_step = 16 
    num_samples = 8 
    max_new_tokens = 512
    temperature = 1.0
    top_k = 50
    unembedding_lr = 0.004
    embedding_lr = 0.2
    matrix_lr = 0.02
    weight_decay = 0.0
    init_lr_frac = 0.05
    num_epochs = 2
    save_every = 250
    eval_every = 50
    eval_examples = 200
    
    # Mini-batch configuration
    mini_batch_size = 8  # Size of each mini-batch
    num_mini_epochs = 1   # Number of passes over the collected data

    use_wandb = True # log wandb
    if use_wandb:
        wandb.init(project="nanochat-rl", name=run_name)
    else:
        wandb_run = DummyWandb()

    # Init compute
    try:
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    except Exception as e:
        print(f"compute_init failed (expected in notebook if not torchrun): {e}")
        print("Falling back to manual device selection.")
        ddp = False
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    master_process = ddp_rank == 0
    pt_dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype)

    print(f"Loading model from {source}...")
    model, tokenizer, meta = load_model(source, device, phase="eval")
    engine = Engine(model, tokenizer)
    print("Model loaded.")

    # Initialize Tasks
    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // examples_per_step) * num_epochs
    print(f"Training examples: {len(train_task)}")
    print(f"Calculated number of steps: {num_steps}")

    # Helper function to collect a batch of rollouts
    @torch.no_grad()
    def collect_rollouts(ddp, device):
        """Collect examples_per_step rollouts with num_samples each"""
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
        iterator = itertools.cycle(rank_indices)
        
        all_inputs = []
        all_targets = []
        all_advantages = []
        
        for _ in range(examples_per_step):
            example_idx = next(iterator)
            
            conversation = train_task[example_idx]
            
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            
            model.eval()
            generated_token_sequences = []
            masks = []
            
            # batch gen so we dont oom
            num_sampling_steps = max(1, num_samples // device_batch_size)
            current_samples = 0
            
            for sampling_step in range(num_sampling_steps):
                remaining = num_samples - current_samples
                batch_size = min(device_batch_size, remaining)
                if batch_size <= 0: break
                
                seed = (example_idx * 1000 + sampling_step) & 0x7FFFFFFF
                
                with autocast_ctx:
                    generated_token_sequences_batch, masks_batch = engine.generate_batch(
                        tokens,
                        num_samples=batch_size,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        seed=seed,
                    )
                generated_token_sequences.extend(generated_token_sequences_batch)
                masks.extend(masks_batch)
                current_samples += batch_size
                
            rewards = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                reward = train_task.reward(conversation, generated_text)
                rewards.append(reward)
                
            max_length = max(len(seq) for seq in generated_token_sequences)
            padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
            padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
            
            ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
            
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_ids[:, 1:] == 0] = -1
            
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            
            # GRPO Advantage Calculation
            mu = rewards.mean()
            advantages = rewards - mu
            
            all_inputs.append(inputs)
            all_targets.append(targets)
            all_advantages.append(advantages)
        
        # Find the maximum sequence length across all rollouts
        max_seq_len = max(inp.size(1) for inp in all_inputs)
        
        # Synchronize max_seq_len across all ranks in DDP
        if ddp:
            max_seq_len_tensor = torch.tensor(max_seq_len, device=device)
            torch.distributed.all_reduce(max_seq_len_tensor, op=torch.distributed.ReduceOp.MAX)
            max_seq_len = max_seq_len_tensor.item()
        
        # Pad all sequences to the same length
        padded_inputs = []
        padded_targets = []
        
        for inputs, targets in zip(all_inputs, all_targets):
            current_len = inputs.size(1)
            if current_len < max_seq_len:
                pad_len = max_seq_len - current_len
                # Pad inputs with assistant_end token
                inputs = torch.cat([inputs, torch.full((inputs.size(0), pad_len), assistant_end, dtype=torch.long, device=device)], dim=1)
                # Pad targets with -1 (masked out)
                targets = torch.cat([targets, torch.full((targets.size(0), pad_len), -1, dtype=torch.long, device=device)], dim=1)
            
            padded_inputs.append(inputs)
            padded_targets.append(targets)
        
        # Concatenate all rollouts and move to CPU to save GPU memory
        all_inputs = torch.cat(padded_inputs, dim=0).cpu()
        all_targets = torch.cat(padded_targets, dim=0).cpu()
        all_advantages = torch.cat(all_advantages, dim=0).cpu()
        
        return all_inputs, all_targets, all_advantages

    # Setup Optimizer
    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"]

    def get_lr_multiplier(it):
        lrm = 1.0 - it / num_steps
        return max(0.0, lrm)

    print("Starting training...")

    # num of steps to run
    steps_to_run = 250

    for step in range(steps_to_run):
        if step % eval_every == 0:
            print(f"--- Eval at step {step} ---")

        # Collect rollouts for this step
        all_inputs, all_targets, all_advantages = collect_rollouts(ddp, device)
        
        total_samples = all_inputs.size(0)
        print(f"Collected {total_samples} samples for step {step}")
        
        # Update learning rate
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        
        # Mini-batch training
        total_loss = 0.0
        num_updates = 0
        
        for mini_epoch in range(num_mini_epochs):
            # Shuffle indices for this mini-epoch (on CPU)
            indices = torch.randperm(total_samples)
            
            for i in range(0, total_samples, mini_batch_size):
                # Get mini-batch indices
                mini_batch_indices = indices[i:i + mini_batch_size]
                
                # Move mini-batch to GPU
                mini_inputs = all_inputs[mini_batch_indices].to(device)
                mini_targets = all_targets[mini_batch_indices].to(device)
                mini_advantages = all_advantages[mini_batch_indices].to(device)
                
                model.train()
                
                with autocast_ctx:
                    # loss_reduction='none' gives (B, T)
                    logp = -model(mini_inputs, mini_targets, loss_reduction='none').view_as(mini_inputs)
                
                # Policy gradient: - sum(log_prob * advantage)
                pg_obj = (logp * mini_advantages.unsqueeze(-1)).sum()
                
                num_valid = (mini_targets >= 0).sum().clamp(min=1)
                pg_obj = pg_obj / num_valid
                
                loss = -pg_obj
                loss.backward()
                
                # Model weight update
                for opt in optimizers:
                    opt.step()
                model.zero_grad(set_to_none=True)
                
                total_loss += loss.item()
                num_updates += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        mean_reward = all_advantages.mean().item() + all_advantages.mean().item()  # Approximate from advantages
        
        print(f"Step {step} | Avg Loss: {avg_loss:.4f} | LR Multiplier: {lrm:.4f} | Updates: {num_updates}")
        
        if use_wandb:
            wandb.log({
                "step": step, 
                "loss": avg_loss, 
                "lrm": lrm,
                "num_updates": num_updates
            })

        # Save Model
        if (step > 0 and step % save_every == 0) or step == steps_to_run - 1:
            base_dir = get_base_dir()
            depth = model.config.n_layer
            model_tag = f"d{depth}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
            model_config_kwargs = model.config.__dict__
            save_checkpoint(
                checkpoint_dir,
                step,
                model.state_dict(),
                None,
                {
                    "model_config": model_config_kwargs,
                }
            )
            print(f"✅ Saved model checkpoint to {checkpoint_dir}")

if __name__ == "__main__":
    main()