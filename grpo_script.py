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
    run_name = "grpo_gsm8k_script" 
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

    use_wandb = True # log wandb
    if use_wandb:
        wandb.init(project="nanochat-rl", name=run_name)
    else:
        wandb_run = DummyWandb()

    # Init compute
    # compute_init handles DDP, but we'll assume rank 0 for notebook simplicity if not launched with torchrun.
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

    # Helper function to get a batch of data
    @torch.no_grad()
    def get_batch():
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
        
        iterator = itertools.cycle(rank_indices)
        
        while True:
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
                # print("Gen text:", generated_text)
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
            
            # GRPO Advantage Calculation: (r - mean(r)) / (std(r) + eps) or just (r - mean(r))
            # The script uses just (r - mu)
            mu = rewards.mean()
            # std = rewards.std()
            advantages = rewards - mu
            # advantages = advantages / (std + 1e-8) # Optional: Normalize
            
            yield generated_token_sequences, inputs, targets, rewards, advantages

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
    batch_iterator = get_batch() 

    # num of steps to run
    steps_to_run = 250

    for step in range(steps_to_run):
        if step % eval_every == 0:
            print(f"--- Eval at step {step} ---")

        rewards_list = []
        
        # calc gradients over the batch size 'examples_per_step'
        
        for example_step in range(examples_per_step):
            sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
            
            model.train()
            
            # inputs_all is (num_samples, seq_len)
            total_samples = inputs_all.size(0)
            num_passes = (total_samples + device_batch_size - 1) // device_batch_size
            
            for pass_idx in range(num_passes):
                b0 = pass_idx * device_batch_size
                b1 = min((pass_idx + 1) * device_batch_size, total_samples)
                
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                advantages = advantages_all[b0:b1]
                
                with autocast_ctx:
                    # loss_reduction='none' gives (B, T)
                    logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
                
                #policy gradient: - sum(log_prob * advantage)
                # mask out invalid tokens (targets == -1)
                pg_obj = (logp * advantages.unsqueeze(-1)).sum()
                
                num_valid = (targets >= 0).sum().clamp(min=1)
                # avg over the total number of examples we are processing in this step
                pg_obj = pg_obj / (num_valid * num_passes * examples_per_step) # normalize objective
                
                loss = -pg_obj
                loss.backward()
                
            rewards_list.append(rewards_all.mean().item())
            
        # model weight updates
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        
        mean_reward = sum(rewards_list) / len(rewards_list)
        print(f"Step {step} | Mean Reward: {mean_reward:.4f} | LR Multiplier: {lrm:.4f}")
        
        if use_wandb:
            wandb.log({"step": step, "reward": mean_reward, "lrm": lrm})

        # 5. Save Model
        if (step > 0 and step % save_every == 0) or step == steps_to_run - 1:
            base_dir = get_base_dir()
            depth = model.config.n_layer
            model_tag = f"d{depth}" # base the model tag on the depth of the base model
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
            model_config_kwargs = model.config.__dict__
            save_checkpoint(
                checkpoint_dir,
                step,
                model.state_dict(),
                None, # note: we don't bother to save the optimizer state
                {
                    "model_config": model_config_kwargs,
                }
            )
            print(f"✅ Saved model checkpoint to {checkpoint_dir}")

if __name__ == "__main__":
    main()
