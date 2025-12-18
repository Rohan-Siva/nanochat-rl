
import sys
import os
import argparse
import torch
from contextlib import nullcontext
from tqdm import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from nanochat.common import compute_init, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K
import torch.distributed as dist
import json
import time

def get_args():
    parser = argparse.ArgumentParser(description="Run inference with NanoChat model")
    parser.add_argument("--source", type=str, default="rl", help="Model source (sft, mid, rl, base)")
    parser.add_argument("--tag", type=str, default=None, help="Model tag (e.g. d32). If not provided, finds largest.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load. If not provided, finds last.")
    parser.add_argument("--device", type=str, default="", help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (float32, bfloat16)")
    # some validation args added
    parser.add_argument("--eval", action="store_true", help="Run evaluation on GSM8K test set")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--pass_at_k", type=int, default=1, help="K for Pass@K metric (number of samples per prompt)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation (to avoid OOM)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")
    return parser.parse_args()

def main():
    args = get_args()

    device_type = args.device if args.device else autodetect_device_type()
    print("Device Type:",device_type)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    print0(f"Using device: {device}")
    
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print0(f"Loading model from source: {args.source}...")
    try:
        model, tokenizer, meta = load_model(
            args.source, 
            device, 
            phase="eval", 
            model_tag=args.tag, 
            step=args.step
        )
        print0("✅ Model loaded successfully.")
    except Exception as e:
        print0(f"❌ Error loading model: {e}")
        return

    engine = Engine(model, tokenizer)
    
    # def tokens
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
        
        #batch process, prevents oom
        num_batches = (num_samples + args.batch_size - 1) // args.batch_size
        
        for b in range(num_batches):
            current_batch_size = min(args.batch_size, num_samples - b * args.batch_size)
            
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

    if args.eval:
        if ddp_rank == 0:
            print0(f"\nEvaluating on GSM8K (test split) - {args.num_samples} samples (Pass@{args.pass_at_k}) across {ddp_world_size} GPUs")
        
        val_task = GSM8K(subset="main", split="test")
        
        correct_count = 0
        total_count = 0
        
        num_eval = min(args.num_samples, len(val_task))
        
        
        # distribute work across ranks, each rank takes every ddp_world_size-th sample starting from ddp_rank
        my_indices = range(ddp_rank, num_eval, ddp_world_size)
        
        if ddp_rank == 0:
            pbar = tqdm(total=num_eval, desc=f"Evaluating Pass@{args.pass_at_k}", unit="sample")
        
        for i in my_indices:
            conversation = val_task.get_example(i)
            prompt = conversation['messages'][0]['content']
            
            if i == 0 and ddp_rank == 0:
                print0(f"\n[Sample Input]:\n{prompt}")
                
                # extract correct answer for logging
                correct_msg = conversation['messages'][-1]['content']
                if isinstance(correct_msg, list):
                    correct_text = "".join([part["text"] for part in correct_msg])
                else:
                    correct_text = str(correct_msg)
                print0(f"\n[Correct Answer]:\n{correct_text}")

            responses = generate_responses(prompt, num_samples=args.pass_at_k, temperature=args.temperature)

            if i == 0 and ddp_rank == 0:
                print0(f"\n[Sample Output (1/{len(responses)})]:\n{responses[0]}\n")
                print0("-" * 50)
            
            # check for pass@k
            any_correct = False
            for response in responses:
                reward = val_task.reward(conversation, response)
                if reward == 1.0:
                    any_correct = True
                    break
            
            if any_correct:
                correct_count += 1
            total_count += 1
            
            # ideally we'd gather progress updates but this is simple enough
            if ddp_rank == 0:
                 pbar.update(ddp_world_size)
                 
        # aggregate results
        device_tensor = torch.tensor([correct_count, total_count], dtype=torch.long, device=device)
        dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        
        global_correct = device_tensor[0].item()
        global_total = device_tensor[1].item()
        
        if ddp_rank == 0:
            pbar.close()
            final_acc = global_correct / global_total if global_total > 0 else 0.0
            print0(f"\nFinal Pass@{args.pass_at_k} Accuracy on {global_total} samples: {final_acc:.4f}")
            
            # output to file now
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"eval_{args.source}_{args.tag or 'auto'}_{args.step or 'last'}_pass{args.pass_at_k}_{timestamp}.json"
            output_path = os.path.join(args.output_dir, filename)
            
            results_data = {
                "config": vars(args),
                "final_accuracy": final_acc,
                "total_samples": global_total,
                "correct_samples": global_correct,
                "timestamp": timestamp,
                "dataset": "GSM8K_test"
            }
            
            with open(output_path, "w") as f:
                json.dump(results_data, f, indent=4)
            print0(f"Results saved to {output_path}")
        
    else:
        print0("\nInteractive Chat Mode. Type 'quit' or 'exit' to stop.\n")
        
        def chat_stream(prompt, temperature=0.6, top_k=50, max_tokens=256):
            conversation_tokens = [bos]
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(prompt))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)
            
            generate_kwargs = {
                "num_samples": 1,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
            }
            
            print(f"\nUser: {prompt}")
            print("Assistant: ", end="", flush=True)
            
            with autocast_ctx:
                for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                    token = token_column[0]
                    if token == assistant_end: 
                        break
                    token_text = tokenizer.decode([token])
                    print(token_text, end="", flush=True)
            print("\n")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                if not user_input.strip():
                    continue
                    
                chat_stream(user_input, temperature=args.temperature)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
