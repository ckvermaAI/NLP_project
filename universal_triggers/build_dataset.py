from datasets import load_dataset, DatasetDict, concatenate_datasets
import re


def add_word_to_hypothesis(example, word):
    if example["hypothesis"][0].isupper():
        example["hypothesis"] = example['hypothesis'][0].lower() + example["hypothesis"][1:]
    example["hypothesis"] = word + " " + example["hypothesis"]
    return example

def topk_triggers_score(log_file_path, k=10):
    """
    Extract unique (<word>, <score>) pairs from a log file and sort them by score.

    Args:
    - log_file_path (str): Path to the log file.
    - k (int): Number of triggers to extract

    Returns:
    - list of tuples: Sorted list of unique (<word>, <score>) pairs.
    """
    trigger_pattern = r"Current Triggers: (\w+),\s+: (\d+\.?\d*)"
    unique_triggers = {}

    with open(log_file_path, 'r') as file:
        for line in file:
            # Search for the pattern in each line
            match = re.search(trigger_pattern, line)
            if match:
                word, score = match.groups()
                score = float(score)  # Convert score to a float for sorting
                # Update the dictionary to keep only one instance of each word
                if word not in unique_triggers or unique_triggers[word] > score:
                    unique_triggers[word] = score

    # Sort the unique_triggers dictionary by score
    sorted_triggers = sorted(unique_triggers.items(), key=lambda x: x[1])
    return [trigger[0] for trigger in sorted_triggers[:k]]

def create_dataset(triggers_logs, repo_name,
                   num_samples = {'train': 3000,
                                  'validation': 3000},
                   seed=28):
    """
    Args: triggers_log (dict): python dictionary with triggers logs for
                               each of the three classes
    Pushes the dataset to HuggingFace
    """
    class_to_label = {"entailment": 0, "neutral": 1, "contradiction": 2}
    snli_dataset = load_dataset("snli")
    print(f"Loading the dataset splits ....")
    dataset = {split: snli_dataset[split].shuffle(seed=seed).select(range(size))
                     for split, size in num_samples.items()}
    print(f"Extracting the triggers ....")
    trigger_tokens = {label_class:topk_triggers_score(path, k=5) for label_class, path in triggers_logs.items()}
    print(f"trigger_tokens: {trigger_tokens}")

    new_dataset = {}
    for split in num_samples:
        # Iterate over train, validation splits
        if split == 'train':
            print(f"Creating augmented data for train split")
            augmented_data = []
            for label_class in trigger_tokens:
                # Iterate over 3 classes of snli dataset
                triggers = trigger_tokens[label_class]
                num_triggers = len(triggers)
                # Original dataset
                chunk = dataset[split].filter(lambda x: x['label'] == class_to_label[label_class.split('_')[0]])
                augmented_data.append(chunk)
                # Augmented dataset
                chunk = chunk.map(lambda example, index: add_word_to_hypothesis(example, word=triggers[index % num_triggers]), with_indices=True)
                augmented_data.append(chunk)

            new_dataset[split] = concatenate_datasets(augmented_data)

        if split == 'validation':
            print(f"Creating augmented data for validation split")
            for label_class in trigger_tokens:
                # Iterate over 3 classes of snli dataset
                triggers = trigger_tokens[label_class]
                num_triggers = len(triggers)
                chunk = dataset[split].filter(lambda x: x['label'] == class_to_label[label_class.split('_')[0]])
                new_dataset[label_class.split('_')[0]] = chunk.shuffle(seed=seed)
                chunk = chunk.map(lambda example, index: add_word_to_hypothesis(example, word=triggers[index % num_triggers]), with_indices=True)
                new_dataset[f"{label_class}"] = chunk.shuffle(seed=seed)

    # Push the modified dataset to your repository
    new_dataset = DatasetDict(new_dataset)
    new_dataset.push_to_hub(repo_name)
    print("Dataset pushed to hugging face")


if __name__ == "__main__":
    triggers_logs = {
        "entailment_triggers": "./universal_triggers/hotflip/entailment-contradiction.log",
        "neutral_triggers": "./universal_triggers/hotflip/neutral-contradiction.log",
        "contradiction_triggers": "./universal_triggers/hotflip/contradiction-neutral.log",
        }
    create_dataset(triggers_logs, repo_name="ckverma/snli_validation")