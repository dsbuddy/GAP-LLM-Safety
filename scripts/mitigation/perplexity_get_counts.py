# Load csv
import pandas as pd
from scipy.stats import ks_2samp, anderson_ksamp, mannwhitneyu
from itertools import combinations

df = pd.read_csv('perplexity_dataset_results.csv')
methods = {}

# Run a count through the dataset by each method and check how many are less than a constant m
for m in [3.0, 4.0, 4.2, 5.0]:
    print("\n{}\n".format(m))
    for method, subset in df.groupby('Method'):
        perplexities = subset['Perplexity'].tolist()
        methods[method] = perplexities
        count_less_than_m = sum(i < m for i in perplexities)
        total_count = len(perplexities)
        print(f"{method}: {count_less_than_m} / {total_count}")

# Get all unique pairs of methods
method_pairs = combinations(methods.keys(), 2)

# Perform pairwise Kolmogorov-Smirnov tests
for method1, method2 in method_pairs:
    data1 = methods[method1]
    data2 = methods[method2]
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = ks_2samp(data1, data2)
    
    # Anderson-Darling test
    ad_statistic, _, ad_pvalue = anderson_ksamp([data1, data2])
    
    # Mann-Whitney U test
    mw_statistic, mw_pvalue = mannwhitneyu(data1, data2, alternative='two-sided')
    
    print(f"Comparing {method1} and {method2}:")
    print(f"Kolmogorov-Smirnov statistic: {ks_statistic:.2f}, p-value: {ks_pvalue:.4f}")
    print(f"Anderson-Darling statistic: {ad_statistic:.2f}, p-value: {ad_pvalue:.4f}")
    print(f"Mann-Whitney U statistic: {mw_statistic:.2f}, p-value: {mw_pvalue:.4f}")
    
    # Interpret the results
    alpha = 0.05  # Significance level
    if ks_pvalue < alpha or ad_pvalue < alpha or mw_pvalue < alpha:
        print("At least one test suggests the distributions are significantly different.")
    else:
        print("The tests do not suggest a significant difference between the distributions.")
    
    print()  # Add a blank line for readability