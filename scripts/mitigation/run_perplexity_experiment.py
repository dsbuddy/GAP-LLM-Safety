import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class PerplexityFilter:
    """
    Filter sequences based on perplexity of the sequence.
    
    Parameters
    ----------
    model : transformers.PreTrainedModel
        Language model to use for perplexity calculation.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer to use for encoding sequences.
    threshold : float
        Threshold for -log perplexity. sequences with perplexity below this threshold
        will be considered "good" sequences.
    window_size : int
        Size of window to use for filtering. If window_size is 10, then the
        -log perplexity of the first 10 tokens in the sequence will be compared to
        the threshold. 
    """
    def __init__(self, model, tokenizer, threshold, window_size=10):
        self.tokenizer = tokenizer
        self.model = model#.cuda()
        self.threshold = threshold
        self.window_threshold = threshold
        self.window_size = window_size
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    def get_log_perplexity(self, sequence):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        # input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
        if torch.cuda.is_available():
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
        else:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt')
        with torch.no_grad():   
            loss = self.model(input_ids, labels=input_ids).loss
        return loss.item()

    def get_max_log_perplexity_of_goals(self, sequences):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        all_loss = []
        cal_log_prob = []
        for sequence in sequences:
            if torch.cuda.is_available():
                input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
            else:
                input_ids = self.tokenizer.encode(sequence, return_tensors='pt')
            with torch.no_grad():   
                output = self.model(input_ids, labels=input_ids)
                loss = output.loss
            all_loss.append(loss.item())
            cal_log_prob.append(self.get_log_prob(sequence).mean().item())
        return max(all_loss)
    
    def get_max_win_log_ppl_of_goals(self, sequences):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        all_loss = []
        for sequence in sequences:
            if torch.cuda.is_available():
                input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
            else:
                input_ids = self.tokenizer.encode(sequence, return_tensors='pt')
            with torch.no_grad():   
                loss = self.model(input_ids, labels=input_ids).loss
            all_loss.append(loss.item())
        
        return max(all_loss)
    
    def get_log_prob(self, sequence):
        """
        Get the log probabilities of the token.

        Parameters
        ----------
        sequence : str
        """
        if torch.cuda.is_available():
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
        else:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(input_ids, labels=input_ids).logits
        logits = logits[:, :-1, :].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        log_probs = self.cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return log_probs
    
    def filter(self, sequences):
        """
        Filter sequences based on log perplexity.

        Parameters
        ----------
        sequences : list of str

        Returns
        -------
        filtered_log_ppl : list of float
            List of log perplexity values for each sequence.
        passed_filter : list of bool
            List of booleans indicating whether each sequence passed the filter.
        """
        filtered_log_ppl = []
        passed_filter = []
        for sequence in sequences:
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            if NLL_by_token.mean() <= self.threshold:
                passed_filter.append(True)
                filtered_log_ppl.append(NLL_by_token.mean().item())
            else:
                passed_filter.append(False)
                filtered_log_ppl.append(NLL_by_token.mean().item())
        return filtered_log_ppl, passed_filter
    
    def filter_window(self, sequences, reverse=False):
        """
        Filter sequences based on log perplexity of a window of tokens.
        
        Parameters
        ----------
        sequences : list of str
            List of sequences to filter.
        reverse : bool
            If True, filter sequences based on the last window_size tokens in the sequence.
            If False, filter sequences based on the first window_size tokens in the sequence.

        Returns
        -------
        filtered_log_ppl_by_window : list of list of float
            List of lists of log perplexity values for each sequence.
        passed_filter_by_window : list of list of bool
            List of lists of booleans indicating whether each sequence passed the filter.
        passed : list of bool
            List of booleans indicating whether each sequence passed the filter.
        """
        filtered_log_ppl_by_window = []
        passed_filter_by_window = []
        passed = []
        for sequence in sequences:
            sequence_window_scores = []
            passed_window_filter = []
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            for i in np.arange(0, len(NLL_by_token), self.window_size):
                if not reverse:
                    window = NLL_by_token[i:i+self.window_size]
                else:
                    if i == 0:
                        window = NLL_by_token[-self.window_size:]
                    elif -(-i-self.window_size) > len(NLL_by_token) and i != 0:
                        window = NLL_by_token[:-i]
                    else:
                        window = NLL_by_token[-i-self.window_size:-i]
                if window.mean() <= self.window_threshold:
                    passed_window_filter.append(True)
                    sequence_window_scores.append(window.mean().item())
                else:
                    passed_window_filter.append(False)
                    sequence_window_scores.append(window.mean().item())
            if all(passed_window_filter):
                passed.append(True)
            else:
                passed.append(False)
            passed_filter_by_window.append(passed_window_filter)
            filtered_log_ppl_by_window.append(sequence_window_scores)
        return filtered_log_ppl_by_window, passed_filter_by_window, passed

# Load the contents of a file into a list of stripped lines
def load_file(filename):
    with open(filename, 'r') as file:
        lines_stripped = [line.strip() for line in file]
    return lines_stripped

# Generate a dataset by combining clean seeds with GCG suffixes
def generate_dataset_advbench_with_gcg_suffixes(clean_seeds, suffixes_filename='perplexity_dataset_gcg_paper_suffixes.csv'):    
    df = pd.read_csv(suffixes_filename)
    suffixes = df['attack_recipe_config'].tolist()
    seeds_with_suffixes = [f"{seed} {suffix}" for seed in clean_seeds for suffix in suffixes]
    return seeds_with_suffixes

# Generate a dataset by combining clean seeds with templates
def generate_dataset_advbench_with_gptfuzzer_templates(clean_seeds, templates_filename='perplexity_dataset_gptfuzzer_paper_templates.csv'):
    df = pd.read_csv(templates_filename)
    templates = df['text'].tolist()
    seeds_with_templates = [template.replace("[INSERT PROMPT HERE]", seed) for template in templates for seed in clean_seeds]
    return seeds_with_templates

# Compute the perplexity values for a given set of sequences using the PerplexityFilter
def compute_perplexity(perplexity_filter, sequences):
    ppl_values, _ = perplexity_filter.filter(sequences)
    return ppl_values

if __name__ == '__main__':
    # Load the data generated by each method
    dataset_advbench_clean_seeds = load_file('perplexity_dataset_advbench_clean_seeds.txt')
    dataset_advbench_with_gcg_suffixes = generate_dataset_advbench_with_gcg_suffixes(dataset_advbench_clean_seeds)
    dataset_advbench_with_gptfuzzer_templates = generate_dataset_advbench_with_gptfuzzer_templates(dataset_advbench_clean_seeds)
    tap_results_file = "./results_advbench_gpt35_tap.json"
    gap_results_file = "./results_advbench_gpt35_gap.json"
    with open(tap_results_file) as json_file_tap:
        dataset_tap_gpt35_advbench = [obj['jailbreak_prompt'] for obj in json.load(json_file_tap)]
    with open(gap_results_file) as json_file_gap:
        dataset_gap_gpt35_advbench = [obj['jailbreak_prompt'] for obj in json.load(json_file_gap)]

    # Load the Llama-2 model and tokenizer for perplexity computation
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')

    # Initialize the PerplexityFilter with the loaded model, tokenizer, and desired threshold and window size
    gc.collect()
    threshold = -1.0  # Set the desired threshold for -log perplexity
    window_size = 10  # Set the desired window size
    perplexity_filter = PerplexityFilter(model, tokenizer, threshold, window_size)

    # Compute perplexity for each dataset
    print("Loading perplexity values for clean seeds...")
    perplexity_values_clean = compute_perplexity(perplexity_filter, dataset_advbench_clean_seeds)
    print("Loading perplexity values for TAP learned jailbreaks...")
    perplexity_values_tap = compute_perplexity(perplexity_filter, dataset_tap_gpt35_advbench)
    print("Loading perplexity values for GAP learned jailbreaks...")
    perplexity_values_gap = compute_perplexity(perplexity_filter, dataset_gap_gpt35_advbench)
    print("Loading perplexity values for GCG Suffixes...")
    perplexity_values_gcg = compute_perplexity(perplexity_filter, dataset_advbench_with_gcg_suffixes)
    print("Loading perplexity values for GPTFuzzer Templates...")
    perplexity_values_gptfuzzer = compute_perplexity(perplexity_filter, dataset_advbench_with_gptfuzzer_templates)

    # Methods
    methods = {
        'AdvBench Clean Seeds': perplexity_values_clean,
        'TAP Learned Jailbreaks': perplexity_values_tap,
        'GAP Learned Jailbreaks': perplexity_values_gap,
        'AdvBench Clean Seeds + GCG Suffix': perplexity_values_gcg,
        'AdvBench Clean Seeds + GPTFuzzer Templates': perplexity_values_gptfuzzer
    }

    # Create a DataFrame for plotting
    data = pd.DataFrame({
       'Perplexity': perplexity_values_clean + perplexity_values_tap + perplexity_values_gap + perplexity_values_gcg + perplexity_values_gptfuzzer,
        'Method': ['AdvBench Clean Seeds'] * len(perplexity_values_clean) +
                  ['TAP Learned Jailbreaks'] * len(perplexity_values_tap) +
                  ['GAP Learned Jailbreaks'] * len(perplexity_values_gap) +
                  ['AdvBench Clean Seeds+ GCG Suffix'] * len(perplexity_values_gcg) +
                  ['AdvBench Clean Seeds+ GPTFuzzer Templates'] * len(perplexity_values_gptfuzzer)
    })

    # Dump dataframe
    data.to_csv('perplexity_dataset_results.csv', index=False)
 
    # Load perplexity results from perplexity_dataset_results.csv
    data = pd.read_csv('perplexity_dataset_results.csv')
    print(data.head())
    
    # Plot the violin plot with different colors for each method
    # plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(8, 6), dpi=300)
    sns.violinplot(x='Method', y='Perplexity', data=data, inner='quartile', palette='tab10')

    plt.title("Distribution of Perplexity across Different Jailbreak Methods", fontsize=16)
    plt.xlabel("Method", fontsize=20)
    plt.xticks(fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Perplexity", fontsize=20)
    plt.savefig('results_perplexity_methods_violin.pdf', dpi=300)

    # Plot the KDE curves
    plt.figure(figsize=(8, 6))
    for method in data['Method'].unique():
        sns.kdeplot(data=data[data['Method'] == method]['Perplexity'], shade=True, label=method)

    # Set the titles and labels
    # plt.title("Distribution of Perplexity across Different Jailbreak Methods", fontsize=18)
    plt.title("Distribution of Perplexity across Different Jailbreak Methods", fontsize=18, loc='left', x=-0.06)
    # Set title and move loc to left
    plt.xlabel("Perplexity", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)    
    # Remove white space on right
    plt.tight_layout()
    plt.savefig('results_perplexity_methods_density.pdf', dpi=300)