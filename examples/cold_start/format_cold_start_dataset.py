import pandas as pd
import json

if __name__ == '__main__':
    dataset = []

    original_seeds = 'cold_start_generated_seeds.json'
    seeds = {}
    with open(original_seeds, 'r') as f_original:
        data_seeds = json.load(f_original)
        for obj_seed in data_seeds:
            seeds[obj_seed['seed-prompt']['generated-seed']] = {
                'category': obj_seed['category'],
                'behavior': obj_seed['behavior'],
            }

    harmful_seeds = 'cold_start_generated_seeds_harmful_all.json'
    with open(harmful_seeds, 'r') as f_harmful:
        data_harmful = json.load(f_harmful)
        for harmful_seed in data_harmful:
            datum = {
                'query': harmful_seed['query'],
                'prompt': harmful_seed['jailbreak_prompt'],
                'category': seeds[harmful_seed['query']]['category'],
                'behavior': seeds[harmful_seed['query']]['behavior'],
                'harmful': 1,
            }
            dataset.append(datum)

    benign_seeds = 'cold_start_generated_seeds_benign_all.json'
    with open(benign_seeds, 'r') as f_benign:
        data_benign = json.load(f_benign)
        for benign_seed in data_benign:
            datum = {
                'query': benign_seed['query'],
                'prompt': benign_seed['query'],
                'category': benign_seed['category'],
                'behavior': benign_seed['behavior'],
                'harmful': 0,
            }
            dataset.append(datum)

    with open('cold_start_dataset_finetune.json', 'w') as f_out:
        json.dump(dataset, f_out, indent=2)

    # Create a pandas DataFrame from the dataset
    df = pd.DataFrame(dataset)

    # Write the DataFrame to a CSV file
    df.to_csv('cold_start_dataset_finetune.csv', index=False)