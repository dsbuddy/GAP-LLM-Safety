# headers = ['Behavior','FunctionalCategory','SemanticCategory','Tags','ContextString','BehaviorID']

import csv
import json
from collections import defaultdict

def combine_csv_json():
    # Load the CSV data
    csv_data = []
    with open('/home/dansw/intern-workspace/src/TAP-enhanced-dansw/src/tap_enhanced_dansw/EasyJailbreak/easyjailbreak/datasets/data/harmbench_behaviors_text_all.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            csv_data.append(row)

    
    # Load the JSON data
    with open('/home/dansw/intern-workspace/src/TAP-enhanced-dansw/src/tap_enhanced_dansw/EasyJailbreak/easyjailbreak/datasets/data/harmbench_targets_text.json', 'r') as json_file:
        json_data = json.load(json_file)
    
    for row in csv_data:
        behavior_id = row['BehaviorID']
        if behavior_id in json_data:
            row['reference_responses'] = list([str(json_data[behavior_id])])
        row['query'] = row.pop('Behavior')

    field_names = ['FunctionalCategory', 'SemanticCategory', 'Tags', 'ContextString', 'BehaviorID', 'reference_responses', 'query']

    with open('harmbench.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(csv_data)


# # Example usage
combined_data = combine_csv_json()
# print(json.dumps(combined_data, indent=4))

