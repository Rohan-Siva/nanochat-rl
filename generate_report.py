import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

def extract_metrics():
    results = []

    json_files = glob.glob('eval_results/*.json') + glob.glob('my_results*/*.json')
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)

        if 'config' in data: # Individual checkpoints
            config = data['config']
            step = config.get('step', data.get('step', None)) # sometimes config misses step if it's 'last'
            pass_k = config.get('pass_at_k', None)
            accuracy = data.get('final_accuracy', None)
            split = config.get('split', 'test')

            # Extract specific runs based on naming
            if 'eval_rl_d32' in jf:
                run_type = 'RL Base (v4/v5)'
            elif 'eval_rl_auto' in jf:
                run_type = 'RL Custom (auto)'
            elif 'eval_sft' in jf:
                run_type = 'SFT Baseline'
            else:
                run_type = 'Other'

            if step is None and 'last' in jf:
                # Approximate last step for RL auto
                step = 250

            results.append({
                'file': jf,
                'run_type': run_type,
                'step': step,
                'pass_k': pass_k,
                'split': split,
                'accuracy': accuracy
            })

        elif 'results' in data or 'metrics' in data: # Batch evaluations
            metrics_list = data.get('results', data.get('metrics', []))
            pass_k = data.get('pass_at_k', 1)

            if 'batch_eval_v5' in jf:
                run_type = 'RL v5 Stable'
            else:
                run_type = 'Batch Eval'

            for m in metrics_list:
                results.append({
                    'file': jf,
                    'run_type': run_type,
                    'step': m.get('step'),
                    'pass_k': pass_k,
                    'split': 'test', # Batch evals are typically test
                    'accuracy': m.get('accuracy')
                })
        
        elif 'pass_at_k' in data and 'result' in data:
            # e.g eval_sft_d32_650_pass1_...
            res = data['result']
            results.append({
                'file': jf,
                'run_type': 'SFT Baseline',
                'step': res.get('step'),
                'pass_k': data['pass_at_k'],
                'split': 'test',
                'accuracy': res.get('accuracy')
            })

    return pd.DataFrame(results)

def generate_plots():
    df = extract_metrics()
    
    # Filter for Pass@1 metrics on test set
    df_p1 = df[(df['pass_k'] == 1) & (df['split'] == 'test')].dropna(subset=['step', 'accuracy'])
    
    plt.figure(figsize=(10, 6))
    
    # Plot SFT Baseline
    sft_data = df_p1[df_p1['run_type'] == 'SFT Baseline']
    if not sft_data.empty:
        sft_acc = sft_data['accuracy'].mean() # If multiple, take mean
        plt.axhline(y=sft_acc, color='r', linestyle='--', label=f'SFT Baseline (Acc: {sft_acc:.3f})')
        
    # Plot RL v5 Stable learning curve
    rl_v5_data = df_p1[df_p1['run_type'] == 'RL v5 Stable'].sort_values(by='step')
    if not rl_v5_data.empty:
        plt.plot(rl_v5_data['step'], rl_v5_data['accuracy'], marker='o', label='GRPO v5 Stable', color='b')
        
    # Plot RL base points (eval_rl_d32)
    rl_base_data = df_p1[df_p1['run_type'] == 'RL Base (v4/v5)'].sort_values(by='step')
    if not rl_base_data.empty:
        plt.scatter(rl_base_data['step'], rl_base_data['accuracy'], label='GRPO Intermediates', color='g', alpha=0.6)

    plt.title('Pass@1 Accuracy on GSM8K Test vs Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Pass@1 Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/grpo_learning_curve.png', dpi=300, bbox_inches='tight')
    print("Saved learning curve plot to assets/grpo_learning_curve.png")
    
    # Plot final Pass@128 comparison
    df_p128 = df[(df['pass_k'] == 128)].dropna(subset=['accuracy'])
    
    if not df_p128.empty:
        plt.figure(figsize=(8, 6))
        
        # Group by run type and split
        summary = df_p128.groupby(['run_type', 'split'])['accuracy'].max().unstack()
        
        summary.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
        plt.title('Pass@128 Accuracy Validation (GRPO vs SFT)')
        plt.ylabel('Pass@128 Accuracy')
        plt.xlabel('Model Setup')
        plt.xticks(rotation=45)
        plt.legend(title='Dataset Split')
        plt.tight_layout()
        
        plt.savefig('assets/pass128_comparison.png', dpi=300)
        print("Saved Pass@128 comparison plot to assets/pass128_comparison.png")

    # Generate markdown tables summary
    print("\n--- Summary Tables ---")
    print("\nPass@1 Accuracy:")
    table_p1 = df_p1.groupby(['run_type', 'step'])['accuracy'].max().reset_index()
    print(table_p1.to_markdown(index=False))
    
    print("\nPass@128 Accuracy:")
    table_p128 = df_p128.groupby(['run_type', 'split', 'step'])['accuracy'].max().reset_index()
    print(table_p128.to_markdown(index=False))

if __name__ == "__main__":
    generate_plots()
