import zipfile
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, accuracy_score,
    roc_auc_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

ZIP_PATH = "/ADFA-LD.zip"
EXTRACTED_PATH = "./ADFA-LD"

if not os.path.exists(EXTRACTED_PATH):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_PATH)
    print("Done.")
else:
    print("Dataset already extracted.")

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


BASE_PATH = "./ADFA-LD"
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "Training_Data_Master")
VALIDATION_DATA_PATH = os.path.join(BASE_PATH, "Validation_Data_Master")
ATTACK_DATA_PATH = os.path.join(BASE_PATH, "Attack_Data_Master")
TERM_SIZE = 3
UNKNOWN_TOKEN = "__UNK__"  # Token for unseen n-grams in the test set

def load_traces_from_dir(directory_path, label):
    all_traces_as_lists = []

    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return [], []

    for root, _, files in sorted(os.walk(directory_path)):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    # List of syscall of the same process
                    content = f.read().strip().split()
                    if content:
                        # List of lists of syscall of different process
                        all_traces_as_lists.append(content)
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")

    labels = [label] * len(all_traces_as_lists)
    return all_traces_as_lists, labels

def create_ngrams(trace_list, n):
    # Combines the elements in the list into a single string, with an underscore added between every two elements
    return ['_'.join(trace_list[i:i+n]) for i in range(len(trace_list) - n + 1)]

# Convert a trace to text with handling for UNK tokens
def transform_trace_to_text(ngrams_list, vocab):
    processed_ngrams = [ngram if ngram in vocab else UNKNOWN_TOKEN for ngram in ngrams_list]
    return ' '.join(processed_ngrams)

def print_results(model_name, y_true, y_pred):
    print(f"{model_name} Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (1, 1):
        if y_true[0] == 0:
            cm_full = np.array([[cm[0][0], 0], [0, 0]])
        else:
            cm_full = np.array([[0, 0], [0, cm[0][0]]])
        cm = cm_full

    print(pd.DataFrame(cm,
                       index=['Actual Normal', 'Actual Attack'],
                       columns=['Predicted Normal', 'Predicted Attack']))

    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC: {auc:.4f}")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)


    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=True,
                xticklabels=['Normal (0)', 'Attack (1)'],
                yticklabels=['Normal (0)', 'Attack (1)'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {model_name}')

    plt.show()
    plt.close()


# Loading All Data
train_traces, train_labels = load_traces_from_dir(TRAIN_DATA_PATH, 0)
validation_traces, validation_labels = load_traces_from_dir(VALIDATION_DATA_PATH, 0)
attack_traces, attack_labels = load_traces_from_dir(ATTACK_DATA_PATH, 1)

all_traces = train_traces + validation_traces + attack_traces
all_labels = np.array(train_labels + validation_labels + attack_labels)

# Splitting Train/Test
train_traces, test_traces, train_labels, test_labels = train_test_split(
    all_traces, all_labels, test_size=0.4, random_state=42, stratify=all_labels
)

# Creating N-Grams
train_ngrams_lists = [create_ngrams(trace, TERM_SIZE) for trace in train_traces]
test_ngrams_lists = [create_ngrams(trace, TERM_SIZE) for trace in test_traces]

# Create vocabulary of unique ngrams of training
training_vocabulary = set()
for ngrams in train_ngrams_lists:
    training_vocabulary.update(ngrams)

# Convert to text with UNK
processed_train_texts = [transform_trace_to_text(ngrams, training_vocabulary) for ngrams in train_ngrams_lists]
processed_test_texts = [transform_trace_to_text(ngrams, training_vocabulary) for ngrams in test_ngrams_lists]


# Create the final feature set including the UNKNOWN_TOKEN
final_feature_set = sorted(list(training_vocabulary) + [UNKNOWN_TOKEN])

# Initialize the vectorizer with the explicit vocabulary
vectorizer = CountVectorizer(vocabulary=final_feature_set)


# Transform the data
X_train_mvsr = vectorizer.fit_transform(processed_train_texts)

# Transform on test data, using the already defined vocabulary
X_test_mvsr = vectorizer.transform(processed_test_texts)
rf_model = RandomForestClassifier(n_estimators=100, criterion="entropy", class_weight="balanced", random_state=42)
rf_model.fit(X_train_mvsr, train_labels)
rf_predictions = rf_model.predict(X_test_mvsr)
print_results("Random Forest", test_labels, rf_predictions)