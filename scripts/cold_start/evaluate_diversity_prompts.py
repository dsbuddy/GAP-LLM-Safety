import json
import math
import nltk
from collections import Counter
import numpy as np
from scipy.stats import zipf
from datasets import load_dataset

import nltk
from nltk.translate.bleu_score import sentence_bleu

def compute_self_bleu(prompts):
    """
    Calculates the Self-BLEU score of a list of prompts.
    
    Args:
        prompts (list): A list of generated prompts.
        
    Returns:
        float: The Self-BLEU score.
    """
    bleu_scores = []
    
    for i, prompt in enumerate(prompts):
        references = [p for j, p in enumerate(prompts) if i != j]
        bleu_score = sentence_bleu(references, prompt)
        bleu_scores.append(bleu_score)
    
    return 1 - sum(bleu_scores) / len(prompts)

# def compute_self_bleu(prompts):
#     """
#     Compute Self-BLEU score for a list of prompts.
#     """
#     bleu_scores = []
#     for i in range(len(prompts)):
#         refs = [prompt for j, prompt in enumerate(prompts) if j != i]
#         hyp = prompts[i]
#         bleu_score = nltk.translate.bleu_score.sentence_bleu(refs, hyp)
#         bleu_scores.append(bleu_score)
#     return sum(bleu_scores) / len(bleu_scores)

def compute_ngram_entropy(prompts, n):
    """
    Compute the entropy of the n-gram distribution for the given prompts.
    """
    all_text = " ".join(prompts)
    ngrams = nltk.ngrams(all_text.split(), n)
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    ngram_probs = [count / total_ngrams for count in ngram_counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in ngram_probs if p > 0)
    return entropy


import numpy as np
from scipy.optimize import curve_fit

def zipf_distribution(k, s):
    """
    Zipf distribution function.
    
    Parameters:
    k (array-like): Ranks of the words.
    s (float): Zipf coefficient.
    
    Returns:
    numpy.ndarray: Probabilities of the words.
    """
    return 1 / (k ** s)

def compute_zipf_coefficient(prompts):
    """
    Compute the Zipf coefficient for a list of generated prompts.
    
    Parameters:
    prompts (list): List of generated prompts.
    
    Returns:
    float: Zipf coefficient.
    """
    # Count the frequency of each word
    word_counts = {}
    for prompt in prompts:
        for word in prompt.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort the words by frequency in descending order
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract the ranks and frequencies
    ranks = np.array([i+1 for i in range(len(sorted_word_counts))])
    frequencies = np.array([count for _, count in sorted_word_counts])
    
    # Fit the Zipf distribution
    popt, _ = curve_fit(zipf_distribution, ranks, frequencies)
    
    return popt[0]

# def compute_zipf_coefficient(prompts):
#     """
#     Compute the Zipf coefficient for the given prompts, assuming the word frequencies follow a Zipfian distribution.
#     """
#     all_text = " ".join(prompts)
#     word_counts = Counter(all_text.split())
#     total_words = sum(word_counts.values())
#     frequencies = np.array([count for count in word_counts.values()])
#     ranks = np.arange(1, len(frequencies) + 1)

#     # Fit the Zipf distribution to the data
#     zipf_dist = zipf(a=2.0)  # Initialize with a valid value greater than 1
#     zipf_dist.a = 1.0 / np.sum((frequencies / total_words) * np.log(ranks))

#     # Ensure the Zipf coefficient is valid
#     if zipf_dist.a <= 1:
#         return np.nan  # Return NaN if the coefficient is invalid

#     # Compute the Zipf coefficient
#     expected_frequencies = zipf_dist.pmf(ranks)
#     observed_frequencies = frequencies / total_words
#     zipf_coeff = np.sum(observed_frequencies * np.log(expected_frequencies / observed_frequencies))

#     return zipf_coeff

import nltk

def compute_unique_ngram_ratio(prompts, n):
    """
    Compute the ratio of unique n-grams to all n-grams in the given prompts.
    """
    all_text = " ".join(prompts)
    ngrams = list(nltk.ngrams(all_text.split(), n))  # Convert to list
    unique_ngrams = set(ngrams)
    total_ngrams = len(ngrams)
    unique_ratio = len(unique_ngrams) / total_ngrams
    return unique_ratio


def compute_diversity_metrics(prompts):
    """
    Compute various diversity metrics for the given prompts.
    """
    self_bleu = compute_self_bleu(prompts)
    ngram_entropy = compute_ngram_entropy(prompts, n=3)
    zipf_coeff = compute_zipf_coefficient(prompts)
    unique_ngram_ratio = compute_unique_ngram_ratio(prompts, n=3)
    return {
        "bleu": self_bleu,
        "entropy": ngram_entropy,
        "zipf": zipf_coeff,
        "unique": unique_ngram_ratio
    }

# Load the contents of a file into a list of stripped lines
def load_file(filename):
    with open(filename, 'r') as file:
        lines_stripped = [line.strip() for line in file]
    return lines_stripped
if __name__ == '__main__':


    # dataset_advbench_clean_seeds = load_file('perplexity_dataset_advbench_clean_seeds.txt')
    # print(compute_diversity_metrics(dataset_advbench_clean_seeds))
    
    # Iterate over all json files in current working directory
    learned_jailbreaks_files = ['results_advbench_gpt35_tap.json', 'results_advbench_gpt35_gap.json', 'results_advbench_llama2_tap.json', 'results_advbench_llama2_gap.json', 'cold_start_generated_seeds_harmful_all.json']
    labels = ['gpt35_tap', 'gpt35_gap', 'llama2_tap', 'llama2_gap', 'cold_start_harmful']

    for i, file in enumerate(learned_jailbreaks_files):
        with open(file) as json_file:
            # dataset = [obj['jailbreak_prompt'] for obj in json.load(json_file)]
            dataset = [obj['jailbreak_prompt'] for obj in json.load(json_file) if obj['call_info']['Jailbreak'] == 1]

        print(f"File: {labels[i]}")
        print(compute_diversity_metrics(dataset))
        print("\n")

        # Save all diversity metrics to dataframe
