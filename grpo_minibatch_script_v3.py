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
    # CONFIGS - V3: Intermediate LRs, GAPO-style advantages, more samples
    # =========================================================================
    run_name = "grpo_gsm8k_minibatch_v6_gapo" 
    source = "sft" 
    dtype = "bfloat16"
    device_batch_size = 4 
    examples_per_step = 16 
    
    # CHANGE #6: More samples per question (16 like chat_rl.py)
    num_samples = 16
    
    max_new_tokens = 512
    temperature = 1.0
    top_k = 50
    
    # CHANGE #1: Intermediate learning rates (5x reduction instead of 10x)
    unembedding_lr = 0.002   # was 0.0004 in v2, 0.004 in v1
    embedding_lr = 0.1       # was 0.02 in v2, 0.2 in v1
    matrix_lr = 0.01         # was 0.002 in v2, 0.02 in v1
    weight_decay = 0.0
    
    # CHANGE #1: Higher init_lr_frac (matches chat_rl.py)
    init_lr_frac = 0.05      # was 0.025 in v2
    
    num_epochs = 1
    save_every = 100
    eval_every = 50
    eval_examples = 200
    
    # CHANGE #3: Reduced mini-epochs, larger mini-batch
    mini_batch_size = 64     # was 32 in v2
    num_mini_epochs = 2      # was 4 in v2, 1 in v1
    clip_eps = 0.2
    
    # Keep KL regularization (not used in loss but tracked)
    kl_beta = 0.01
    
    # CHANGE #7: Higher gradient clipping threshold
    max_grad_norm = 5.0      # was 1.0 in v2
    
    # Keep advantage clipping but less aggressive
    adv_clip = 10.0          # was 5.0 in v2

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
            "init_lr_frac": init_lr_frac,
            "adv_clip": adv_clip,
        })
    pt_dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=pt_dtype)

    print(f"Loading model from {source}...")
    model, tokenizer, meta = load_model(source, device, phase="eval")
    engine = Engine(model, tokenizer)
    print("Model loaded.")

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

    # Rollout collection with GAPO-style advantages
    @torch.no_grad()
    def collect_rollouts(ddp, device):
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        
        all_inputs = []
        all_targets = []
        all_advantages = []
        all_rewards = []
        all_old_log_probs = []
        
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
            
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            
            # CHANGE #2: GAPO-style advantages (r - mu), no std division
            # This matches chat_rl.py and avoids gradient collapse when std is tiny
            mu = rewards.mean()
            advantages = rewards - mu
            
            # Clip advantages to prevent extreme gradients (less aggressive)
            advantages = torch.clamp(advantages, -adv_clip, adv_clip)
            
            with autocast_ctx:
                nll = model(inputs, targets, loss_reduction='none').view_as(inputs)
                old_log_probs = -nll

            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())
            all_advantages.append(advantages.cpu())
            all_rewards.append(rewards.cpu())
            all_old_log_probs.append(old_log_probs.cpu())
        
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
        all_old_log_probs = torch.cat(padded_old_log_probs, dim=0)
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

    # Shorter warmup (20 steps instead of 50)
    warmup_steps = min(20, num_steps // 10)
    
    def get_lr_multiplier(it):
        if it < warmup_steps:
            # Linear warmup
            return it / warmup_steps
        else:
            # Cosine decay to 10% of peak
            progress = (it - warmup_steps) / max(1, num_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    print("Starting training...")
    print(f"  - Learning rates: emb={embedding_lr}, matrix={matrix_lr}, unemb={unembedding_lr}")
    print(f"  - Init LR fraction: {init_lr_frac}")
    print(f"  - GAPO-style advantages (r - mu, no std division)")
    print(f"  - Num samples per question: {num_samples}")
    print(f"  - Gradient clipping max norm: {max_grad_norm}")
    print(f"  - Mini-batch size: {mini_batch_size}, mini-epochs: {num_mini_epochs}")
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
        
        # CHANGE #8: Track advantage statistics
        adv_mean = all_advantages.mean().item()
        adv_std = all_advantages.std().item()
        adv_min = all_advantages.min().item()
        adv_max = all_advantages.max().item()
        
        # Update learning rate
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        
        # Mini-batch training with token-level PPO
        total_loss = 0.0
        num_updates = 0
        
        # CHANGE #8: Track gradient norms and ratio statistics
        total_grad_norm = 0.0
        ratio_means = []
        ratio_maxs = []
        
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
                    micro_advantages = all_advantages[micro_batch_indices].to(device)
                    micro_old_log_probs = all_old_log_probs[micro_batch_indices].to(device)
                    
                    # Create mask for valid (non-padding) tokens
                    valid_mask = (micro_targets >= 0).float()
                    
                    model.train()
                    
                    with autocast_ctx:
                        logp = -model(micro_inputs, micro_targets, loss_reduction='none').view_as(micro_inputs)
                    
                    # Compute token-level ratio in float32
                    logp_f32 = logp.float()
                    old_logp_f32 = micro_old_log_probs.float()
                    ratio = torch.exp(logp_f32 - old_logp_f32)
                    
                    # CHANGE #8: Track ratio statistics
                    with torch.no_grad():
                        valid_ratio = ratio[valid_mask > 0]
                        if valid_ratio.numel() > 0:
                            ratio_means.append(valid_ratio.mean().item())
                            ratio_maxs.append(valid_ratio.max().item())
                    
                    # Mask ratio by valid tokens
                    ratio = ratio * valid_mask
                    
                    # Token-level PPO with sequence-level advantages broadcast to tokens
                    surr1 = ratio * micro_advantages.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * micro_advantages.unsqueeze(-1)
                    
                    pg_obj = torch.min(surr1, surr2).sum()
                    num_valid = valid_mask.sum().clamp(min=1)
                    pg_obj = pg_obj / num_valid
                    
                    loss = -pg_obj
                    
                    # Scale for gradient accumulation
                    loss_scale = len(micro_batch_indices) / current_mini_batch_size
                    scaled_loss = loss * loss_scale
                    
                    scaled_loss.backward()
                    
                    total_loss += loss.item() * loss_scale

                # CHANGE #8: Compute and track gradient norm before clipping
                all_params = []
                for opt in optimizers:
                    for group in opt.param_groups:
                        all_params.extend(group['params'])
                
                grad_norm = nn_utils.clip_grad_norm_(all_params, max_grad_norm)
                total_grad_norm += grad_norm.item()
                
                # Update model weights after mini batch
                for opt in optimizers:
                    opt.step()
                
                num_updates += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_updates if num_updates > 0 else 0.0
        mean_reward = all_rewards.mean().item()
        
        # CHANGE #8: Compute ratio statistics
        avg_ratio_mean = sum(ratio_means) / len(ratio_means) if ratio_means else 1.0
        max_ratio = max(ratio_maxs) if ratio_maxs else 1.0
        
        # Sync metrics across all ranks
        if ddp:
            metrics = torch.tensor([avg_loss, mean_reward, avg_grad_norm, adv_mean, adv_std], dtype=torch.float, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_loss = metrics[0].item()
            mean_reward = metrics[1].item()
            avg_grad_norm = metrics[2].item()
            adv_mean = metrics[3].item()
            adv_std = metrics[4].item()
        
        if master_process:
            print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lrm:.4f} | MeanRwd: {mean_reward:.4f} | GradNorm: {avg_grad_norm:.2f} | AdvMean: {adv_mean:.3f} | AdvStd: {adv_std:.3f}")
            
            if use_wandb:
                wandb.log({
                    "step": step, 
                    "loss": avg_loss,
                    "lrm": lrm,
                    "num_updates": num_updates,
                    "mean_reward": mean_reward,
                    # CHANGE #8: Log additional diagnostics
                    "grad_norm": avg_grad_norm,
                    "advantage/mean": adv_mean,
                    "advantage/std": adv_std,
                    "advantage/min": adv_min,
                    "advantage/max": adv_max,
                    "ratio/mean": avg_ratio_mean,
                    "ratio/max": max_ratio,
                })

        # Save Model
        if (step > 0 and step % save_every == 0) or step == steps_to_run - 1:
            base_dir = get_base_dir()
            depth = model.config.n_layer
            model_tag = f"d{depth}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints_v6", model_tag)
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
