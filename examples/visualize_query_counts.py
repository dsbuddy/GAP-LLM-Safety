import json
from statistics import stdev
import matplotlib.pyplot as plt

# Load JSON data from a file
def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Extract call counts and queries from the data
def extract_call_counts(data):
    call_counts = []
    queries = []
    jailbreak_count = 0
    for obj in data:
        call_info = obj.get('call_info', {})
        jailbreak_count += call_info.get('Jailbreak', 0)
        counts = [
            call_info.get('Calls to attack_model.generate', 0),
            call_info.get('Calls to eval_model.generate', 0),
            call_info.get('Calls to target_model.generate', 0),
            call_info.get('Calls to attack_model.generate', 0) + call_info.get('Calls to eval_model.generate', 0) + call_info.get('Calls to target_model.generate', 0)
        ]
        call_counts.append(counts)
        queries.append(obj.get('query', 'Unknown Query'))
    print(f"Jailbreak Count: {jailbreak_count}/{len(data)}")
    return call_counts, queries

# Visualize average call counts
def visualize_avg_counts(all_call_counts, labels, image_name, highlight_label=None, call_types=['Attack LLM', 'Evaluation LLM', 'Target LLM', 'All LLM Combined']):
    x_pos = range(len(call_types))
    bar_width = 0.8 / len(all_call_counts)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, call_counts in enumerate(all_call_counts):
        avg_counts = [sum(counts) / len(call_counts) for counts in zip(*call_counts)]
        std_counts = [stdev(counts) for counts in zip(*call_counts)]
        c_counts = [sum(counts) for counts in zip(*call_counts)]
        total_counts = [sum(c_counts)]
        print(f"Label [{labels[i]}]: Average = {avg_counts}")
        print(f"Label [{labels[i]}]: Standard Deviation = {std_counts}")
        print(f"Label [{labels[i]}]: Counts = {c_counts}")
        print(f"Label [{labels[i]}]: Total Counts = {total_counts}")
        bars = ax.bar([x + i * bar_width for x in x_pos], avg_counts, width=bar_width, label=labels[i])
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            if labels[i] == highlight_label:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, '*', ha='center', va='bottom', fontsize=14)

    ax.set_xlabel('LLM Model to Query')
    ax.set_ylabel('Average Number of Queries')
    ax.set_xticks([x + (len(all_call_counts) - 1) * bar_width / 2 for x in x_pos])
    ax.set_xticklabels(call_types, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(image_name)
    print("Saved")
    plt.close()

if __name__ == "__main__":
    # model_name = 'gpt35'
    model_name = 'llama2'
    file_paths = [f'results_advbench_{model_name}_tap.json', f'results_advbench_{model_name}_gap.json']
    labels = ['TAP', 'GAP']
    image_name = f'results_query_count_{model_name}.png'
    all_data = [load_data(file_path) for file_path in file_paths]
    all_call_counts, all_queries = zip(*[extract_call_counts(data) for data in all_data])
    visualize_avg_counts(all_call_counts, labels, image_name, highlight_label='Max')