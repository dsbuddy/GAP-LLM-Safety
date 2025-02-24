import json
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def extract_call_counts(data):
    call_counts = []
    queries = []
    for obj in data:
        call_info = obj.get('call_info', {})
        counts = [
            call_info.get('Calls to attack_model.generate', 0),
            call_info.get('Calls to eval_model.generate', 0),
            call_info.get('Calls to target_model.generate', 0),
            sum(count for count in call_info.values() if 'Calls to' in count)
        ]
        call_counts.append(counts)
        queries.append(obj.get('query', 'Unknown Query'))
    return call_counts, queries

def plot_query_budget(query_budgets, jailbreak_counts_tap, jailbreak_counts_gap, model_name, llm):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(query_budgets, jailbreak_counts_gap, marker='o', color='b', label='GAP')
    ax.plot(query_budgets, jailbreak_counts_tap, marker='o', color='r', label='TAP')
    ax.set_xlabel('Query Budget')
    ax.set_ylabel('Number of Jailbreaks')
    ax.set_yticks(range(0, 51, 5))
    ax.grid(True)
    ax.set_title(f'Graph of Attacks # Jailbreaks per {llm} Query Budget [{model_name}]')
    ax.legend(loc='upper left')
    plt.savefig(f'results_query_budget_model_{model_name}_llm_{llm}.png')

def count_jailbreaks(all_data, query_budget, is_tap, llm_type):
    jailbreak_count = 0
    for i, model_data in enumerate(all_data):
        for seed in model_data:
            calls = seed['call_info'].get(f'Calls to {llm_type}.generate', 0)
            is_jailbreak = seed['eval_results'][0]
            if i == 0:
                if (calls <= query_budget) and (is_tap and is_jailbreak == 1):
                    jailbreak_count += 1
            else:
                if (calls <= query_budget) and (not is_tap and is_jailbreak == 1):
                    jailbreak_count += 1
    return jailbreak_count

def count_and_append_jailbreaks(all_data, query_budgets, llm_type):
    tap_jailbreak_counts = []
    gap_jailbreak_counts = []

    for query_budget in query_budgets:
        tap_jailbreak_count = count_jailbreaks(all_data, query_budget, True, llm_type)
        gap_jailbreak_count = count_jailbreaks(all_data, query_budget, False, llm_type)
        tap_jailbreak_counts.append(tap_jailbreak_count)
        gap_jailbreak_counts.append(gap_jailbreak_count)

    return tap_jailbreak_counts, gap_jailbreak_counts

def main():
    model_name = 'gpt35'
    # model_name = 'llama2'
    file_paths = [f'results_advbench_{model_name}_tap.json', f'results_advbench_{model_name}_gap.json']

    all_data = [load_data(file_path) for file_path in file_paths]
    query_budgets = [10, 15, 20, 23, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]

    jailbreak_counts_tap_target, jailbreak_counts_gap_target = count_and_append_jailbreaks(all_data, query_budgets, 'target_model')
    jailbreak_counts_tap_eval, jailbreak_counts_gap_eval = count_and_append_jailbreaks(all_data, query_budgets, 'eval_model')
    jailbreak_counts_tap_attack, jailbreak_counts_gap_attack = count_and_append_jailbreaks(all_data, query_budgets, 'attack_model')

    plot_query_budget(query_budgets, jailbreak_counts_tap_target, jailbreak_counts_gap_target, model_name, 'Target LLM')
    plot_query_budget(query_budgets, jailbreak_counts_tap_attack, jailbreak_counts_gap_attack, model_name, 'Attacker LLM')
    plot_query_budget(query_budgets, jailbreak_counts_tap_eval, jailbreak_counts_gap_eval, model_name, 'Evaluator LLM')

if __name__ == '__main__':
    main()