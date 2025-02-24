import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from datasets import load_dataset

# Loads the original seeds from the JailbreakBench dataset and organizes them into a dictionary where the keys are the categories, and the values are lists of behaviors for each category.
def load_original_seeds():
    original_seeds = {}
    dataset = load_dataset('JailbreakBench/JBB-Behaviors', 'behaviors')
    for seed in dataset['harmful']:
        category = seed['Category']
        behavior = seed['Behavior']
        if category in original_seeds:
            original_seeds[category].append(behavior)
        else:
            original_seeds[category] = [behavior]
    return original_seeds

# Loads results from a JSON file and organizes them by category and specified field.
def load_results(results_json, result_field):
    loaded_results = {}
    with open(results_json, 'r') as f:
        results = json.load(f)
        for result in results:
            if result['category'] in loaded_results:
                loaded_results[result['category']].append(result[result_field])
            else:
                loaded_results[result['category']] = [result[result_field]]
    return loaded_results
       
# Calculates the diversity score (mean pairwise cosine distance) of a list of behaviors using TF-IDF vectorization.
def calculate_diversity(behaviors):
    # Convert behaviors to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(behaviors)
    # Calculate pairwise distances
    distances = squareform(pdist(vectors.toarray(), metric='cosine'))
    # Calculate diversity score (e.g., mean pairwise distance)
    diversity = np.mean(distances)
    return diversity

# Compares the diversity scores of original and generated behaviors for a given category and prints the results.
def compare_diversity(original_behaviors, generated_behaviors, category):
    # Convert behaviors to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    all_behaviors = original_behaviors + generated_behaviors
    all_vectors = vectorizer.fit_transform(all_behaviors)
    # Split the vectors into original and generated
    original_vectors = all_vectors[:len(original_behaviors)]
    generated_vectors = all_vectors[len(original_behaviors):]
    # Calculate pairwise distances
    original_distances = squareform(pdist(original_vectors.toarray(), metric='cosine'))
    generated_distances = squareform(pdist(generated_vectors.toarray(), metric='cosine'))
    # Calculate diversity scores (e.g., mean pairwise distance)
    original_diversity = np.mean(original_distances)
    generated_diversity = np.mean(generated_distances)
    print(f"Category: {category} | Original [{len(original_behaviors)}]= {original_diversity.round(4)} | Generated [{len(generated_behaviors)}]= {generated_diversity.round(4)}")
 
# Performs bootstrapping by resampling the generated behaviors, calculating the diversity score, and returning the mean diversity score over multiple iterations.
def bootstrap_diversity(original_behaviors, generated_behaviors, num_resamples=10000):
    resampled_diversities = []
    for _ in range(num_resamples):
        # Resample the generated dataset by randomly selecting samples
        resampled_data = list(np.random.choice(generated_behaviors, len(original_behaviors), replace=False))
        # Combine original and generated behaviors
        all_behaviors = original_behaviors + resampled_data
        # Convert behaviors to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        all_vectors = vectorizer.fit_transform(all_behaviors)
        # Split the vectors into original and generated
        original_vectors = all_vectors[:len(original_behaviors)]
        generated_vectors = all_vectors[len(original_behaviors):]
        # Calculate pairwise distances
        original_distances = squareform(pdist(original_vectors.toarray(), metric='cosine'))
        generated_distances = squareform(pdist(generated_vectors.toarray(), metric='cosine'))
        # Calculate diversity scores (e.g., mean pairwise distance)
        original_diversity = np.mean(original_distances)
        generated_diversity = np.mean(generated_distances)
        resampled_diversities.append(generated_diversity)
    return np.mean(resampled_diversities)

# Calculates the diversity scores of original and generated behaviors using separate TF-IDF vectorizers to mitigate potential bias and prints the results.
def bias_mitigation_diversity(original_behaviors, generated_behaviors, category):
    # Convert behaviors to TF-IDF vectors using separate vectorizers
    original_vectorizer = TfidfVectorizer()
    original_vectors = original_vectorizer.fit_transform(original_behaviors)
    generated_vectorizer = TfidfVectorizer()
    generated_vectors = generated_vectorizer.fit_transform(generated_behaviors)
    # Calculate pairwise distances
    original_distances = squareform(pdist(original_vectors.toarray(), metric='cosine'))
    generated_distances = squareform(pdist(generated_vectors.toarray(), metric='cosine'))
    # Calculate diversity scores (e.g., mean pairwise distance)
    original_diversity = np.mean(original_distances)
    generated_diversity = np.mean(generated_distances)
    print(f"Category: {category} | Original [{len(original_behaviors)}]= {original_diversity.round(4)} | Generated [{len(generated_behaviors)}]= {generated_diversity.round(4)}")


if __name__ == '__main__':
    # Load seeds from JBB and from generated seeds
    original_seeds = load_original_seeds()
    generated_seeds = load_results('cold_start_generated_seeds.json', 'behavior')

    # Iterate across category
    for category in generated_seeds:
        if category in original_seeds:
            original_behaviors = original_seeds[category]
            generated_behaviors = generated_seeds[category]
            
            # Compare diversity
            compare_diversity(original_behaviors, generated_behaviors, category)
    
            # Bootstrapping
            bootstrapped_diversity = bootstrap_diversity(original_behaviors, generated_behaviors)
            original_diversity = calculate_diversity(original_behaviors)
            print(f"Original dataset diversity: {original_diversity}")
            print(f"Mean resampled dataset diversity: {bootstrapped_diversity}")

            # # Bias mitigation - separate vectorizers
            # bias_mitigation_diversity(original_behaviors, generated_behaviors, category)