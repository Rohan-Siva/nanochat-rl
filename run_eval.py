
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
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    return parser.parse_args()

def main():
    args = get_args()

    device_type = args.device if args.device else autodetect_device_type()
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

    def generate_response(prompt, temperature=0.6, top_k=50, max_tokens=512):
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
        
        response_text = ""
        with autocast_ctx:
            for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
                token = token_column[0]
                if token == assistant_end[0]: 
                    break
                token_text = tokenizer.decode([token])
                response_text += token_text
                
        return response_text

    if args.eval:
        print0(f"\nEvaluating on GSM8K (test split) - {args.num_samples} samples")
        val_task = GSM8K(subset="main", split="test")
        
        correct_count = 0
        total_count = 0
        
        num_eval = min(args.num_samples, len(val_task))
        
        pbar = tqdm(range(num_eval), desc="Evaluating", unit="sample")
        
        for i in pbar:
            conversation = val_task.get_example(i)
            prompt = conversation['messages'][0]['content']
            
            response = generate_response(prompt, temperature=args.temperature)
            
            reward = val_task.reward(conversation, response)
            
            if reward == 1.0:
                correct_count += 1
            total_count += 1
            
            current_acc = correct_count / total_count
            pbar.set_postfix({"accuracy": f"{current_acc:.4f}"})
            
        final_acc = correct_count / total_count
        print0(f"\nFinal Accuracy on {total_count} samples: {final_acc:.4f}")
        
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
                    if token == assistant_end[0]: 
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
