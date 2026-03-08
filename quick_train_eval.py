"""Quick script to evaluate v2 checkpoint on train set to verify training metrics."""
import os
import torch
from nanochat.common import autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

device_type = autodetect_device_type()
device = torch.device(device_type)
ptdtype = torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model from v2 checkpoint
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints_v2", "d32")
print(f"Loading from: {checkpoint_dir}")

# Load using sft as base, then load the checkpoint weights
model, tokenizer, meta = load_model("sft", device, phase="eval")

# Load the v2 checkpoint weights
step = 466
checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
print(f"Loading checkpoint: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

engine = Engine(model, tokenizer)

# Evaluate on train split
train_task = GSM8K(subset="main", split="train")
num_samples = 100  # Evaluate on 100 train examples
pass_at_k = 1
temperature = 1.0
max_tokens = 512
top_k = 50
batch_size = 8

correct = 0
total = 0

print(f"\nEvaluating on {num_samples} training examples with Pass@{pass_at_k}...")
for i in range(num_samples):
    conversation = train_task[i]
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)
    
    with autocast_ctx:
        generated_seqs, masks = engine.generate_batch(
            tokens,
            num_samples=pass_at_k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    any_correct = False
    for seq in generated_seqs:
        generated_tokens = seq[prefix_length:]
        generated_text = tokenizer.decode(generated_tokens)
        reward = train_task.reward(conversation, generated_text)
        if reward == 1.0:
            any_correct = True
            break
    
    if any_correct:
        correct += 1
    total += 1
    
    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/{num_samples}, Accuracy so far: {correct/total:.4f}")

print(f"\nFinal Train Pass@{pass_at_k} Accuracy: {correct}/{total} = {correct/total:.4f}")
