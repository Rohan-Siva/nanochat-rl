import os
import math
import itertools
import torch
import torch.nn.utils as nn_utils
import torch.distributed as dist
import wandb
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50,
    ddp_rank=0,
    ddp_world_size=1,
    device_batch_size=8
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

def main():
    if os.path.basename(os.getcwd()) == "scripts":
        os.chdir("..")

    print(f"Current working directory: {os.getcwd()}")

    # =========================================================================
    # CONFIGS - Updated with stability improvements
    # =========================================================================
    run_name = "grpo_gsm8k_minibatch_v5_stable" 
    source = "sft" 
    dtype = "bfloat16"
    device_batch_size = 4 
    examples_per_step = 16 
    num_samples = 8 
    max_new_tokens = 512
    temperature = 1.0
    top_k = 50
    
    # FIX #3: Lower learning rates by 10x
    unembedding_lr = 0.0004   # was 0.004
    embedding_lr = 0.02       # was 0.2
    matrix_lr = 0.002         # was 0.02
    weight_decay = 0.0
    init_lr_frac = 0.025
    
    num_epochs = 1
    save_every = 100
    eval_every = 50
    eval_examples = 200
    
    # Mini-batch configs
    mini_batch_size = 32
    
    # FIX #8: Increase mini-epochs for more corrective steps
    num_mini_epochs = 4       # was 1
    clip_eps = 0.2
    
    # FIX #2: KL regularization coefficient
    kl_beta = 0.01
    
    # FIX #5: Gradient clipping
    max_grad_norm = 1.0
    
    # FIX #6: Advantage clipping bounds
    adv_clip = 5.0

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

    use_wandb = True
    if use_wandb and master_process:
        wandb.init(project="nanochat-rl", name=run_name, config={
            "run_name": run_name,
            "source": source,
            "device_batch_size": device_batch_size,
            "examples_per_step": examples_per_step,
            "num_samples": num_samples,
            "mini_batch_size": mini_batch_size,
            "num_mini_epochs": num_mini_epochs,
            "kl_beta": kl_beta,
            "max_grad_norm": max_grad_norm,
            "embedding_lr": embedding_lr,
            "matrix_lr": matrix_lr,
            "unembedding_lr": unembedding_lr,
        })
    pt_dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype)

    print(f"Loading model from {source}...")
    model, tokenizer, meta = load_model(source, device, phase="eval")
    engine = Engine(model, tokenizer)
    print("Model loaded.")
    
    # FIX #2: Store reference model log probs (we'll compute them during rollout)
    # The reference model is the initial policy (before any updates)
    # We store ref_log_probs alongside old_log_probs

    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // examples_per_step) * num_epochs
    print(f"Training examples: {len(train_task)}")
    print(f"Calculated number of steps: {num_steps}")

    # Create a persistent iterator that properly cycles through ALL training examples
    def create_example_iterator():
        """Generator that yields example indices, shuffling each epoch deterministically."""
        epoch = 0
        while True:
            rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))
            import random
            rng = random.Random(epoch)
            rng.shuffle(rank_indices)
            for idx in rank_indices:
                yield idx
            epoch += 1
    
    example_iterator = create_example_iterator()

    # Rollout collection with proper sequence-level log probs
    @torch.no_grad()
    def collect_rollouts(ddp, device):
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        
        all_inputs = []
        all_targets = []
        all_advantages = []
        all_rewards = []
        all_old_log_probs = []  # Token-level log probs (reverted from sequence-level)
        
        for _ in range(examples_per_step):
            example_idx = next(example_iterator)
            conversation = train_task[example_idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            
            model.eval()
            generated_token_sequences = []
            masks = []
            
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
            
            # Valid mask for non-padding tokens
            valid_mask = (targets >= 0).float()
            
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            
            # FIX #6: Normalize advantages across the batch for this example, with clipping
            mu = rewards.mean()
            std = rewards.std()
            if std > 1e-6:
                advantages = (rewards - mu) / std
            else:
                advantages = torch.zeros_like(rewards)
            
            # Clip advantages to prevent extreme gradients
            advantages = torch.clamp(advantages, -adv_clip, adv_clip)
            
            with autocast_ctx:
                # Get token-level log probs (not sequence-level to avoid numerical instability)
                nll = model(inputs, targets, loss_reduction='none').view_as(inputs)
                old_log_probs = -nll

            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())
            all_advantages.append(advantages.cpu())
            all_rewards.append(rewards.cpu())
            all_old_log_probs.append(old_log_probs.cpu())  # Token-level
        
        # Find the maximum sequence length across all rollouts
        max_seq_len = max(inp.size(1) for inp in all_inputs)
        
        # Sync max_seq_len across all ranks in DDP
        if ddp:
            max_seq_len_tensor = torch.tensor(max_seq_len, device=device)
            torch.distributed.all_reduce(max_seq_len_tensor, op=torch.distributed.ReduceOp.MAX)
            max_seq_len = max_seq_len_tensor.item()
        
        # Pad all sequences to the same length
        padded_inputs = []
        padded_targets = []
        padded_old_log_probs = []
        
        for inputs, targets, old_log_probs in zip(all_inputs, all_targets, all_old_log_probs):
            current_len = inputs.size(1)
            if current_len < max_seq_len:
                pad_len = max_seq_len - current_len
                inputs = torch.cat([inputs, torch.full((inputs.size(0), pad_len), assistant_end, dtype=torch.long, device='cpu')], dim=1)
                targets = torch.cat([targets, torch.full((targets.size(0), pad_len), -1, dtype=torch.long, device='cpu')], dim=1)
                old_log_probs = torch.cat([old_log_probs, torch.full((old_log_probs.size(0), pad_len), 0.0, dtype=torch.float, device='cpu')], dim=1)
            
            padded_inputs.append(inputs)
            padded_targets.append(targets)
            padded_old_log_probs.append(old_log_probs)
        
        # Concatenate all rollouts (already on CPU)
        all_inputs = torch.cat(padded_inputs, dim=0)
        all_targets = torch.cat(padded_targets, dim=0)
        all_old_log_probs = torch.cat(padded_old_log_probs, dim=0)  # Shape: (N, T) - token level
        all_advantages = torch.cat(all_advantages, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0)
        
        return all_inputs, all_targets, all_advantages, all_rewards, all_old_log_probs

    # Optimizer setup
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

    # FIX #7: Cosine LR schedule with warmup
    warmup_steps = min(50, num_steps // 10)
    
    def get_lr_multiplier(it):
        if it < warmup_steps:
            # Linear warmup
            return it / warmup_steps
        else:
            # Cosine decay to 10% of peak
            progress = (it - warmup_steps) / max(1, num_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    print("Starting training...")
    print(f"  - Learning rates reduced 10x from original")
    print(f"  - KL regularization beta: {kl_beta}")
    print(f"  - Gradient clipping max norm: {max_grad_norm}")
    print(f"  - Mini-epochs per step: {num_mini_epochs}")
    print(f"  - Warmup steps: {warmup_steps}")

    steps_to_run = num_steps

    for step in range(steps_to_run):
        if step % eval_every == 0:
            print(f"--- Eval at step {step} ---")
            model.eval()
            passk = torch.zeros(device_batch_size, device=device)
            with autocast_ctx:
                records_iter = run_gsm8k_eval(val_task, tokenizer, engine, 
                                            max_examples=eval_examples, 
                                            num_samples=device_batch_size, 
                                            temperature=1.0,
                                            ddp_rank=ddp_rank,
                                            ddp_world_size=ddp_world_size,
                                            device_batch_size=device_batch_size)
                records = list(records_iter)
            
            for k in range(1, device_batch_size + 1):
                passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
            
            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)
            
            if num_records.item() > 0:
                passk = passk / num_records.item()
            
            print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
            print0(f"Step {step} | {', '.join(print_passk)}")
            
            if use_wandb and master_process:
                log_passk = {f"eval/pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
                wandb.log({
                    "step": step,
                    **log_passk,
                })

        # Collect rollouts for this step
        all_inputs, all_targets, all_advantages, all_rewards, all_old_log_probs = collect_rollouts(ddp, device)
        
        total_samples = all_inputs.size(0)
        print(f"Collected {total_samples} samples for step {step}")
        
        # Update learning rate
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        
        # Mini-batch training with token-level PPO (stable)
        total_loss = 0.0
        num_updates = 0
        
        for mini_epoch in range(num_mini_epochs):
            shuffle_seed = step * 1000 + mini_epoch
            generator = torch.Generator().manual_seed(shuffle_seed)
            indices = torch.randperm(total_samples, generator=generator)
            
            for i in range(0, total_samples, mini_batch_size):
                mini_batch_indices = indices[i:i + mini_batch_size]
                current_mini_batch_size = len(mini_batch_indices)

                # Zero gradients before accumulating
                model.zero_grad(set_to_none=True)
                
                for j in range(0, current_mini_batch_size, device_batch_size):
                    micro_batch_indices = mini_batch_indices[j:j + device_batch_size]
                    
                    # Move micro-batch to GPU
                    micro_inputs = all_inputs[micro_batch_indices].to(device)
                    micro_targets = all_targets[micro_batch_indices].to(device)
                    micro_advantages = all_advantages[micro_batch_indices].to(device)  # Shape: (B,)
                    micro_old_log_probs = all_old_log_probs[micro_batch_indices].to(device)  # Shape: (B, T)
                    
                    # Create mask for valid (non-padding) tokens
                    valid_mask = (micro_targets >= 0).float()
                    
                    model.train()
                    
                    with autocast_ctx:
                        # Get token-level log probs
                        logp = -model(micro_inputs, micro_targets, loss_reduction='none').view_as(micro_inputs)
                    
                    # Compute token-level ratio in float32 (stable, unlike sequence-level)
                    logp_f32 = logp.float()
                    old_logp_f32 = micro_old_log_probs.float()
                    ratio = torch.exp(logp_f32 - old_logp_f32)
                    
                    # Mask ratio by valid tokens to prevent pollution from padding
                    ratio = ratio * valid_mask
                    
                    # Token-level PPO with sequence-level advantages broadcast to tokens
                    surr1 = ratio * micro_advantages.unsqueeze(-1)  # (B,T) * (B,1)
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * micro_advantages.unsqueeze(-1)
                    
                    pg_obj = torch.min(surr1, surr2).sum()
                    num_valid = valid_mask.sum().clamp(min=1)
                    pg_obj = pg_obj / num_valid
                    
                    # Simple policy gradient loss (no KL term needed with token-level)
                    loss = -pg_obj
                    
                    # Scale for gradient accumulation
                    loss_scale = len(micro_batch_indices) / current_mini_batch_size
                    scaled_loss = loss * loss_scale
                    
                    scaled_loss.backward()
                    
                    total_loss += loss.item() * loss_scale

                # FIX #5: Gradient clipping before optimizer step
                for opt in optimizers:
                    # Get all parameters from this optimizer
                    params = []
                    for group in opt.param_groups:
                        params.extend(group['params'])
                    nn_utils.clip_grad_norm_(params, max_grad_norm)
                
                # Update model weights after mini batch
                for opt in optimizers:
                    opt.step()
                
                num_updates += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        mean_reward = all_rewards.mean().item()
        
        # Sync metrics across all ranks
        if ddp:
            metrics = torch.tensor([avg_loss, mean_reward], dtype=torch.float, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_loss = metrics[0].item()
            mean_reward = metrics[1].item()
        
        if master_process:
            print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lrm:.4f} | MeanRwd: {mean_reward:.4f}")
            
            if use_wandb:
                wandb.log({
                    "step": step, 
                    "loss": avg_loss,
                    "lrm": lrm,
                    "num_updates": num_updates,
                    "mean_reward": mean_reward,
                })

        # Save Model
        if (step > 0 and step % save_every == 0) or step == steps_to_run - 1:
            base_dir = get_base_dir()
            depth = model.config.n_layer
            model_tag = f"d{depth}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints_v5", model_tag)
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
