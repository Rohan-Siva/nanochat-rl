#!/usr/bin/env python
"""
Evaluation script for SFT checkpoint.
Evaluates the SFT model on GSM8K test set.
"""
import os
import sys
import torch
import json
import time
from contextlib import nullcontext
from tqdm import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from nanochat.common import compute_init, autodetect_device_type, print0, get_base_dir
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K
import torch.distributed as dist


def evaluate_checkpoint(checkpoint_dir, step, device, autocast_ctx, 
                        num_samples=300, pass_at_k=1, batch_size=16, 
                        temperature=0.6, ddp=False, ddp_rank=0, ddp_world_size=1):
    """Evaluate a single checkpoint and return accuracy."""
    
    # Load model
    print0(f"\n{'='*60}")
    print0(f"Loading SFT checkpoint step {step} from {checkpoint_dir}")
    print0(f"{'='*60}")
    
    try:
        model, tokenizer, meta = load_model_from_dir(
            os.path.dirname(checkpoint_dir),  # parent dir containing d32
            device,
            phase="eval",
            model_tag=os.path.basename(checkpoint_dir),  # d32
            step=step
        )
    except Exception as e:
        print0(f"❌ Error loading checkpoint: {e}")
        return None
    
    engine = Engine(model, tokenizer)
    
    # Tokenizer setup
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    
    def generate_responses(prompt, num_samples=1, temperature=0.6, top_k=50, max_tokens=512):
        conversation_tokens = [bos]
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(prompt))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        
        all_response_texts = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for b in range(num_batches):
            current_batch_size = min(batch_size, num_samples - b * batch_size)
            
            generate_kwargs = {
                "num_samples": current_batch_size,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
            
            batch_response_texts = [""] * current_batch_size
            finished = [False] * current_batch_size
            
            with autocast_ctx:
                for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                    all_finished = True
                    for i, token in enumerate(token_column):
                        if finished[i]:
                            continue
                        
                        if token == assistant_end: 
                            finished[i] = True
                        else:
                            token_text = tokenizer.decode([token])
                            batch_response_texts[i] += token_text
                            all_finished = False 
                    
                    if all_finished:
                        break
            
            all_response_texts.extend(batch_response_texts)
                
        return all_response_texts
    
    # Evaluate
    val_task = GSM8K(subset="main", split="test")
    
    correct_count = 0
    total_count = 0
    num_eval = min(num_samples, len(val_task))
    
    my_indices = range(ddp_rank, num_eval, ddp_world_size)
    
    if ddp_rank == 0:
        pbar = tqdm(total=num_eval, desc=f"SFT Step {step} Pass@{pass_at_k}", unit="sample")
    
    for i in my_indices:
        conversation = val_task.get_example(i)
        prompt = conversation['messages'][0]['content']
        
        responses = generate_responses(prompt, num_samples=pass_at_k, temperature=temperature)
        
        any_correct = False
        for response in responses:
            reward = val_task.reward(conversation, response)
            if reward == 1.0:
                any_correct = True
                break
        
        if any_correct:
            correct_count += 1
        total_count += 1
        
        if ddp_rank == 0:
            pbar.update(ddp_world_size)
    
    # Aggregate results
    device_tensor = torch.tensor([correct_count, total_count], dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
    
    global_correct = device_tensor[0].item()
    global_total = device_tensor[1].item()
    
    if ddp_rank == 0:
        pbar.close()
    
    accuracy = global_correct / global_total if global_total > 0 else 0.0
    
    # Cleanup model from GPU
    del model, engine
    torch.cuda.empty_cache()
    
    return {
        "step": step,
        "accuracy": accuracy,
        "correct": global_correct,
        "total": global_total,
        "pass_at_k": pass_at_k
    }


def main():
    # Configuration - SFT checkpoint
    checkpoint_base = "/home/jl77863/.cache/nanochat/chatsft_checkpoints"
    model_tag = "d32"
    checkpoint_dir = os.path.join(checkpoint_base, model_tag)
    
    # SFT checkpoint step
    step = 650
    
    # Eval settings
    num_samples = 1000  # Full evaluation
    pass_at_k = 1  # Pass@1 for main metric
    batch_size = 8
    temperature = 0.6
    
    # Init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    ptdtype = torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    print0(f"\n{'#'*60}")
    print0(f"SFT CHECKPOINT EVALUATION")
    print0(f"{'#'*60}")
    print0(f"Checkpoint dir: {checkpoint_dir}")
    print0(f"Step to evaluate: {step}")
    print0(f"Samples: {num_samples}")
    print0(f"Pass@K: {pass_at_k}")
    print0(f"GPUs: {ddp_world_size}")
    print0(f"{'#'*60}\n")
    
    result = evaluate_checkpoint(
        checkpoint_dir, step, device, autocast_ctx,
        num_samples=num_samples, pass_at_k=pass_at_k, 
        batch_size=batch_size, temperature=temperature,
        ddp=ddp, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size
    )
    
    if result and ddp_rank == 0:
        print0(f"\n{'='*60}")
        print0("SFT EVALUATION RESULT")
        print0(f"{'='*60}")
        print0(f"Step {step}: Pass@{pass_at_k} = {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
        print0(f"{'='*60}\n")
        
        # Save results to file
        output_dir = "eval_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"eval_sft_d32_{step}_pass1_{timestamp}.json")
        
        summary = {
            "checkpoint_dir": checkpoint_dir,
            "num_samples": num_samples,
            "pass_at_k": pass_at_k,
            "temperature": temperature,
            "result": result,
            "timestamp": timestamp
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)
        print0(f"📁 Results saved to {output_path}")


if __name__ == "__main__":
    main()
