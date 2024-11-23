import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the file path to the JSONL file
file_path = "/software/users/ckverma/workspace/NLP_project/electra/eval_original_dataset/eval_predictions.jsonl"  # Replace with your file's actual path
# file_path = "/software/users/ckverma/workspace/NLP_project/electra/eval_modified_dataset/eval_predictions.jsonl"  # Replace with your file's actual path

# Initialize lists for true labels and predicted labels
true_labels = []
predicted_labels = []

# Read the JSONL file line by line
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line.strip())  # Parse JSON data from each line
        true_labels.append(data['label'])
        predicted_labels.append(data['predicted_label'])

# Generate the confusion matrix
labels = [0, 1, 2]  # Assuming labels are 0, 1, and 2
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
print(conf_matrix)
