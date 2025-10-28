import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Data
try:
    file_path = 'LMD-2023 [1.75M Elements][Labelled]checked.csv'
    df = pd.read_csv(file_path, low_memory=False)
    print("Reading dataset done!")
    non_helping_columns = ["Version2","SchemaVersion","Level","Opcode","Correlation","TargetProcessId"]

except FileNotFoundError:
    print(f"Error: The file was not found. Please update the 'file_path' variable with the correct location of your LMD-2023 CSV file.")
    X, y = (None, None)

# Preprocess
df = df.drop(columns=['Unnamed: 0.2'], errors='ignore')  # Remove unnecessary debug column

# Prepare labels
df['Label'] = df['Label'].apply(lambda x: 1 if x == 2 else x)  # Convert label 2 to 1 (attack)
y = df['Label']
X = df.drop(columns=['Label', 'SystemTime'])  # Remove label and timestamp columns

# Drop non-numeric and non-helping columns
non_numeric_columns = X.select_dtypes(exclude=['number']).columns.tolist()
X = X.drop(columns=non_numeric_columns)
X = X.dropna(axis=1, how='all')
X = X.drop(columns=non_helping_columns)

print(f"Data shape: {X.shape}, Labels: {y.unique()}")

# Run for dataset LMD-2023
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Initialize and train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight=None,
    criterion="gini"
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]  # Probability for class 1

# Calculate evaluation metrics
auc = roc_auc_score(y_test, y_scores)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("\n--- Evaluation Metrics ---")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False,
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Random Forest')
plt.show()