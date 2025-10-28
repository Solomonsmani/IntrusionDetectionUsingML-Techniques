import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load Data
try:
    file_path = 'LMD-2023 [1.75M Elements][Labelled]checked.csv'
    df = pd.read_csv(file_path, low_memory=False)
    print("Reading dataset done!")
    non_helping_columns = ["TargetProcessId","SchemaVersion","Version2","Correlation","Opcode","Level"]

except FileNotFoundError:
    print(f"Error: The file was not found. Please update the 'file_path' variable with the correct location of your LMD-2023 CSV file.")
    X, y = (None, None)

# Preprocess
df = df.drop(columns=['Unnamed: 0.2'], errors='ignore')  # Remove unnecessary debug column

# Convert all attack labels to 1
df['Label'] = df['Label'].apply(lambda x: 1 if x == 2 else x)
y = df['Label']
X = df.drop(columns=['Label', 'SystemTime'])  # Remove label and original timestamp columns

# Drop non-numeric and non-helping columns
non_numeric_columns = X.select_dtypes(exclude=['number']).columns.tolist()
X = X.drop(columns=non_numeric_columns)
X = X.dropna(axis=1, how='all')
X = X.drop(columns=non_helping_columns)


print(f"Data shape: {X.shape}, Unique labels: {y.unique()}")

# Training and prediction
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=1.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='logloss',
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute metrics
auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"AUC Score: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - XGBoost')
plt.tight_layout()
plt.show()