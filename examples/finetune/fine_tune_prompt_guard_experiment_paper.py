import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def get_class_probabilities(text, temperature=1.0, device=None):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.

    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = inputs.to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def get_jailbreak_score(text, temperature=1.0, device=None):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The probability of the text containing malicious content.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = get_class_probabilities(text, temperature, device)
    return probabilities[0, 2].item()


def get_indirect_injection_score(text, temperature=1.0, device=None):
    """
    Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
    Appropriate for filtering third party inputs (e.g. web searches, tool outputs) into an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The combined probability of the text containing malicious or embedded instructions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = get_class_probabilities(text, temperature, device)
    return (probabilities[0, 1] + probabilities[0, 2]).item()


def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The training dataset.
        pd.DataFrame: The validation dataset.
        pd.DataFrame: The test dataset.
    """
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(filepath)

    # Preprocess data
    # query,prompt,category,behavior,harmful
    X = data["query"].tolist()
    y = data["harmful"].tolist()
    category = data["category"].tolist()

    data = pd.DataFrame({"text": X, "label": y, "category": category})

    # Split into training, validation, and test sets
    train_dataset, test_dataset = train_test_split(
        data, test_size=0.2, random_state=42
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.1, random_state=42
    )

    train_dataset.reset_index(drop=True)
    test_dataset.reset_index(drop=True)
    val_dataset.reset_index(drop=True)

    return train_dataset, val_dataset, test_dataset


def evaluate_batch(
    model, texts, batch_size=32, positive_label=1, temperature=1.0, device=None
):
    """
    Evaluate the model on a batch of texts with temperature-adjusted softmax.

    Args:
        model (transformers.PreTrainedModel): The model to evaluate.
        texts (list of str): The input texts to classify.
        batch_size (int): The number of texts to process in each batch.
        positive_label (int): The label of a multi-label classifier to treat as a positive class.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to run the model on ('cpu', 'cuda', 'mps', etc).
              If None, will use 'cuda' if available, otherwise 'cpu'.

    Returns:
        list of float: The probabilities of the positive class adjusted by the temperature for each text.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert the Pandas Series to a list of strings
    texts = list(texts)

    # Prepare the data loader
    encoded_texts = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    dataset = torch.utils.data.TensorDataset(
        encoded_texts["input_ids"], encoded_texts["attention_mask"]
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    scores = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids, attention_mask = [b.to(device) for b in batch]
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        scaled_logits = logits / temperature
        probabilities = softmax(scaled_logits, dim=-1)
        positive_class_probabilities = probabilities[:, positive_label].cpu().numpy()
        scores.extend(positive_class_probabilities)

    return scores



def plot_roc_curve(labels, scores, filename="roc_curve.png"):
    """Plots the Receiver Operating Characteristic curve."""
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(filename)


def plot_score_distribution(positive_scores, negative_scores, filename="score_dist.png"):
    """Plots the distribution of scores for positive and negative examples."""
    plt.figure(figsize=(10, 6))
    # Plotting positive scores
    sns.kdeplot(
        positive_scores,
        fill=True,
        bw_adjust=0.1,  # specify bandwidth here
        color="darkblue",
        label="Positive",
    )
    # Plotting negative scores
    sns.kdeplot(
        negative_scores,
        fill=True,
        bw_adjust=0.1,  # specify bandwidth here
        color="darkred",
        label="Negative",
    )
    # Adding legend, title, and labels
    plt.legend(prop={"size": 16}, title="Scores")
    plt.title("Score Distribution for Positive and Negative Examples")
    plt.xlabel("Score")
    plt.ylabel("Density")
    # Display the plot
    plt.savefig(filename)


def train_model(
    train_dataset, model, tokenizer, batch_size=32, epochs=1, lr=4e6, device=None
):
    """
    Train the model on the given dataset.

    Args:
        train_dataset (datasets.Dataset): The training dataset.
        model (transformers.PreTrainedModel): The model to train.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the texts.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
        device (str): The device to run the model on ('cpu' or 'cuda').
    """
    # Adjust the model's classifier to have two output labels
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.num_labels = 2

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def collate_fn(batch):
        # Ensure indices are within the valid range
        valid_indices = [i for i in batch if i < len(train_dataset)]
        texts = [train_dataset.iloc[i]["text"] for i in valid_indices]
        labels = torch.tensor([int(train_dataset.iloc[i]["label"]) for i in valid_indices])
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        return encodings.input_ids, encodings.attention_mask, labels

    data_loader = DataLoader(
        train_dataset.index, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Average loss in epoch {epoch + 1}: {total_loss / len(data_loader)}")


def evaluate_metrics(labels, scores, threshold=0.5):
    """
    Calculates and prints accuracy, F1, precision, and recall.

    Args:
        labels (list): True labels.
        scores (list): Predicted scores.
        threshold (float): Threshold for classification.
    """
    predictions = [1 if score >= threshold else 0 for score in scores]
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    # Compute TPR
    cm = confusion_matrix(labels, predictions)
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) else 0
    print(f"True Positive Rate: {tpr:.4f}")

    # Compute FPR
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) else 0
    print(f"False Positive Rate: {fpr:.4f}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")




def load_and_preprocess_dataset(data_source, data_type="csv", rename_col=None, label_col=None, label_condition=None):
    """Generic function to load and preprocess a dataset.

    Args:
      data_source: Path to the dataset file or Hugging Face dataset identifier.
      data_type: Type of data file ("csv" or "jsonl").
      rename_col: Column to rename (optional).
      label_col: Column to use for labels (optional).
      label_condition: Condition for assigning labels (optional).

    Returns:
      df: Pandas DataFrame with the processed data.
    """
    if data_type == "csv":
        df = pd.read_csv(data_source)
    elif data_type == "jsonl":
        df = pd.read_json(data_source, lines=True)
    elif data_type == "parquet":
        df = pd.read_parquet(data_source)
    else:
        raise ValueError("Invalid data_type. Must be 'csv' or 'jsonl'.")

    if rename_col:
        df = df.rename(columns={rename_col: 'text'})
    if label_col and label_condition:
        df['label'] = np.where(df[label_col].apply(label_condition), 1, 0)  # Apply condition to each value
    return df

# --- Evaluation Functions ---

def evaluate_and_report(model, dataset, dataset_name="Test Dataset", temperature=3.0):
    """Evaluates the model on a given dataset and prints the results.

    Args:
        model (transformers.PreTrainedModel): The model to evaluate.
        dataset: Pandas DataFrame containing the dataset.
        dataset_name: Name of the dataset for printing results.
        temperature: Temperature parameter for the evaluation.
    """
    scores = evaluate_batch(model, list(dataset["text"]), positive_label=1, temperature=temperature)
    labels = [int(elt) for elt in dataset["label"]]
    print(f"Metrics for {dataset_name}:")
    evaluate_metrics(labels, scores)


def plot_evaluation_results(dataset, scores, filename_prefix):
    """Plots ROC curve and score distribution.

    Args:
        dataset: Pandas DataFrame containing the dataset.
        scores: List of model scores.
        filename_prefix: Prefix for the filenames of the generated plots.
    """
    labels = [int(elt) for elt in dataset["label"]]

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{filename_prefix}_roc_curve.png")
    plt.close()

    # Plot score distribution
    positive_scores = [scores[i] for i in range(len(dataset)) if labels[i] == 1]
    negative_scores = [scores[i] for i in range(len(dataset)) if labels[i] == 0]
    plt.figure()
    plt.hist(positive_scores, bins=50, alpha=0.5, label='Positive')
    plt.hist(negative_scores, bins=50, alpha=0.5, label='Negative')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend(loc="upper right")
    plt.savefig(f"{filename_prefix}_score_distribution.png")
    plt.close()

if __name__ == '__main__':

    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prompt_injection_model_name = "meta-llama/Prompt-Guard-86M"
    tokenizer = AutoTokenizer.from_pretrained(prompt_injection_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        prompt_injection_model_name
    )
    model.to(device)  # Move the model to the GPU

    # --- Load and Preprocess Data ---
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data("cold_start_dataset_finetune.csv")

    second_dataset = load_and_preprocess_dataset(
        "hf://datasets/lmsys/toxic-chat/data/1123/toxic-chat_annotation_test.csv",
        rename_col="user_input",
        label_col="jailbreaking",
        label_condition=lambda x: x == 1
    )

    # third_dataset = load_and_preprocess_dataset(
    #     "hf://datasets/deepset/prompt-injections/data/test-00000-of-00001-701d16158af87368.parquet",
    #     data_type="parquet",
    # )


    ds = load_dataset("mmathys/openai-moderation-api-evaluation")
    texts = []
    labels = []
    for sample in ds['train']:
        keys = ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
        # Check if any of the keys in the sample are greater than 0
        try:
            if any(sample[key] > 0 for key in keys):
                labels.append(1)
                texts.append(sample['prompt'])
            else:
                labels.append(0)
                texts.append(sample['prompt'])
        except:
            pass

    third_dataset = pd.DataFrame({'text': texts, 'label': labels})
    # # Load the second dataset
    # second_dataset = load_and_preprocess_dataset(
    #     "hf://datasets/jackhhao/jailbreak-classification/balanced/jailbreak_dataset_test_balanced.csv", 
    #     rename_col='prompt', 
    #     label_col='type', 
    #     label_condition=lambda x: x == 'jailbreak'
    # )

    # # Load the third dataset
    # third_dataset = load_and_preprocess_dataset(
    #     "hf://datasets/PKU-Alignment/BeaverTails/round0/30k/test.jsonl.gz",
    #     data_type="jsonl",
    #     rename_col="prompt",
    #     label_col="is_safe", 
    #     label_condition=lambda x: x is True
    # )


    # --- Initial Evaluation ---
    evaluate_and_report(model, test_dataset, "Test Dataset")
    test_scores = evaluate_batch(model, list(test_dataset["text"]), positive_label=1, temperature=3.0) 
    plot_evaluation_results(test_dataset, test_scores, "fig_test_initial")
    evaluate_and_report(model, second_dataset, "Second Dataset")
    evaluate_and_report(model, third_dataset, "Third Dataset")


    # --- Train the Model ---
    train_model(train_dataset, model, tokenizer)

    # --- Save the Model ---
    model_save_path = "prompt_guard_finetuned_4e6.pth" 
    torch.save(model.state_dict(), model_save_path) 

    # --- Load the Saved Model ---
    loaded_model = AutoModelForSequenceClassification.from_pretrained(prompt_injection_model_name) 
    loaded_model.classifier = torch.nn.Linear(loaded_model.classifier.in_features, 2) 
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.to(device)

    # --- Evaluation after Training ---
    evaluate_and_report(loaded_model, test_dataset, "Test Dataset (After Training)") 
    test_scores_loaded = evaluate_batch(loaded_model, test_dataset["text"], positive_label=1, temperature=3.0)
    plot_evaluation_results(test_dataset, test_scores_loaded, "fig_test_trained")
    evaluate_and_report(loaded_model, second_dataset, "Second Dataset (After Training)")
    evaluate_and_report(loaded_model, third_dataset, "Third Dataset (After Training)") 