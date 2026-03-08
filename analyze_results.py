import os
import json
import glob

def analyze_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"\n--- {file_path} ---")
    
    # check keys
    print("Keys:", list(data.keys()))
    if 'config' in data:
        print("Config:", data['config'])
    if 'results' in data or 'metrics' in data:
        metrics = data.get('results', data.get('metrics', {}))
        print("Metrics:", metrics)
    else:
        # maybe the pass@k is at the top level
        pass_metrics = {k: v for k, v in data.items() if 'pass' in k.lower()}
        if pass_metrics:
            print("Pass metrics:", pass_metrics)
        else:
            print("First 5 items:", {k: data[k] for k in list(data.keys())[:5]})

json_files = glob.glob('eval_results/*.json') + glob.glob('my_results*/*.json')
for jf in json_files:
    analyze_json(jf)
