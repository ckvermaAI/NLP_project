import math
from collections import Counter, defaultdict
from datasets import load_dataset
from tqdm import tqdm

# Step 1: Load the SNLI dataset from Hugging Face
snli = load_dataset("snli")

# Step 2: Define utility functions to compute PMI
def compute_word_class_counts(dataset, label_key="label", text_keys=("premise", "hypothesis")):
    """
    Compute word-class joint and marginal counts with add-100 smoothing.
    """
    word_class_counts = Counter()  # Joint counts of (word, class)
    word_counts = Counter()        # Marginal counts of words
    class_counts = Counter()       # Marginal counts of classes
    total_count = 0                # Total count of words

    for example in tqdm(dataset, desc="Processing dataset"):
        label = example[label_key]
        if label == -1:  # Skip if label is invalid
            continue
        text = " ".join([example[key] for key in text_keys])
        words = text.lower().split()  # Tokenize text (basic whitespace tokenizer)

        class_counts[label] += len(words)  # Count words in the current label
        for word in words:
            word_class_counts[(word, label)] += 1
            word_counts[word] += 1
        total_count += len(words)

    # Apply add-100 smoothing
    smoothing = 100
    for word, label in word_class_counts.keys():
        word_class_counts[(word, label)] += smoothing
    for word in word_counts.keys():
        word_counts[word] += smoothing * len(class_counts)
    for label in class_counts.keys():
        class_counts[label] += smoothing * len(word_counts)

    total_count += smoothing * len(word_counts) * len(class_counts)
    return word_class_counts, word_counts, class_counts, total_count

def compute_pmi(word_class_counts, word_counts, class_counts, total_count):
    """
    Compute PMI values for each word-class pair.
    """
    pmi = {}
    for (word, label), joint_count in word_class_counts.items():
        p_word = word_counts[word] / total_count
        p_class = class_counts[label] / total_count
        p_word_class = joint_count / total_count

        pmi[(word, label)] = math.log(p_word_class / (p_word * p_class), 2)
    return pmi

# Step 3: Compute word-class PMI
# Process only the train split of SNLI
train_data = snli["train"]
word_class_counts, word_counts, class_counts, total_count = compute_word_class_counts(train_data)

pmi_scores = compute_pmi(word_class_counts, word_counts, class_counts, total_count)

# Step 4: Print top 10 words (with PMI) for every class
class_top_words = defaultdict(list)
for (word, label), pmi_value in pmi_scores.items():
    class_top_words[label].append((word, pmi_value))

print("Top 10 PMI scores for each class:")
for label, word_pmi_list in class_top_words.items():
    print(f"\nClass {label}:")
    top_10 = sorted(word_pmi_list, key=lambda x: -x[1])[:10]
    for word, pmi_value in top_10:
        print(f"  Word: '{word}', PMI: {pmi_value:.4f}")
