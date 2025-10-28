import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import warnings
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Preprocessing
def preprocess_data(df, features_to_process, label_column_name='Label'):
    df = df.copy()
    features_to_select = [
        'Computer',
        'DestinationPortName',
        'EventID',
        'EventRecordID',
        'Execution_ProcessID',
        'Initiated',
        'ProcessId',
        'SourceIsIpv6',
        'SystemTime',
        'Label'
    ]

    for col in features_to_select:
        if col not in df.columns:
            print(f"Error: {col} column not found in the dataset.")
    df = df[features_to_select]

    # Process SystemTime
    if 'SystemTime' in df.columns:
        df['SystemTime'] = pd.to_datetime(df['SystemTime'], errors='coerce')
        df['SystemTime_year'] = df['SystemTime'].dt.year.astype(str)
        df['SystemTime_month'] = df['SystemTime'].dt.month.astype(str)
        df['SystemTime_week'] = df['SystemTime'].dt.isocalendar().week.astype(str)
        df['SystemTime_day'] = df['SystemTime'].dt.day
        df['SystemTime_hour'] = df['SystemTime'].dt.hour
        df['SystemTime_minute'] = df['SystemTime'].dt.minute
        df['SystemTime_day_of_week'] = df['SystemTime'].dt.dayofweek.astype(str)
        df = df.drop('SystemTime', axis=1)

    # One-Hot Encoding
    ohe_features = [col for col, method in features_to_process.items() if method == 'OHE' and col in df.columns]
    minmax_features = [col for col, method in features_to_process.items() if method == 'MinMax' and col in df.columns]

    if ohe_features:
        encoder = OneHotEncoder(sparse_output=False)
        ohe_array = encoder.fit_transform(df[ohe_features])
        ohe_df = pd.DataFrame(ohe_array, columns=encoder.get_feature_names_out(ohe_features), index=df.index)
        df = df.drop(columns=ohe_features)
        df = pd.concat([df, ohe_df], axis=1)

    if minmax_features:
        scaler = MinMaxScaler()
        df[minmax_features] = scaler.fit_transform(df[minmax_features])

    # Label encoding
    if label_column_name in df.columns:
        df[label_column_name] = df[label_column_name].apply(lambda x: 0 if str(x).strip() == '0' else 1)

    return df

# The 15 selected features along with the preprocessing method applied to each one
features_to_process = {
    'Computer': 'OHE',
    'DestinationPortName': 'OHE',
    'EventID': 'OHE',
    'EventRecordID': 'MinMax',
    'Execution_ProcessID': 'MinMax',
    'Initiated': 'OHE',
    'ProcessId': 'MinMax',
    'SourceIsIpv6': 'OHE',
    'SystemTime_year': 'OHE',
    'SystemTime_month': 'OHE',
    'SystemTime_week': 'OHE',
    'SystemTime_day': 'MinMax',
    'SystemTime_hour': 'MinMax',
    'SystemTime_minute': 'MinMax',
    'SystemTime_day_of_week': 'OHE'
}

# Load Data
try:
    file_path = 'LMD-2023 [1.75M Elements][Labelled]checked.csv'
    df = pd.read_csv(file_path, low_memory=False)
    print("Reading dataset done!")

    # Preprocess Data
    processed_df = preprocess_data(df, features_to_process)
    X = processed_df.drop('Label', axis=1)
    y = processed_df['Label']

except FileNotFoundError:
    print(f"Error: The file was not found. Please update the 'file_path' variable with the correct location of your LMD-2023 CSV file.")
    X, y = (None, None)

if X is not None:
    # Define model with parameters from Table 4 of the paper
    model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', leaf_size=50,  metric='minkowski')


    # Stratified K-Fold Cross-Validation
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    summary_rows = []

    metrics = {"AUC": [], "F1": [], "Precision": [], "Recall": [], "Accuracy": [], "Time": []}
    conf_matrix_sum = np.zeros((2,2))

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        try:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit on normal data only
            normal_X_train = X_train[y_train == 0]
            model.fit(normal_X_train)
            distances, _ = model.kneighbors(X_test)
            anomaly_scores = np.mean(distances, axis=1)
            # Set threshold
            threshold = np.percentile(anomaly_scores, 100 - (100 * y.mean()))
            y_pred = np.where(anomaly_scores > threshold, 1, 0)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            conf_matrix_sum += cm
            print(f"Confusion Matrix:\n{cm}")

            # False samples
            false_positives = X_test[(y_test == 0) & (y_pred == 1)]
            false_negatives = X_test[(y_test == 1) & (y_pred == 0)]

        except Exception as e:
            print(f"Error in fold {fold}: {e}")

        # Calculate and store metrics
        auc_score = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        metrics["AUC"].append(auc_score)
        metrics["F1"].append(f1)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["Accuracy"].append(accuracy)

        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        conf_matrix_sum += cm

        print(f"fold number: {fold}   AUC: {auc_score:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")

    avg_auc = np.mean(metrics['AUC'])
    avg_f1 = np.mean(metrics['F1'])
    avg_precision = np.mean(metrics['Precision'])
    avg_recall = np.mean(metrics['Recall'])
    avg_accuracy = np.mean(metrics['Accuracy'])

    print(f"\nAverage metrics for KNN over {n_splits} folds:")
    print(f"AUC: {avg_auc:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f}")

    # Shows the confusion matrix of the average
    avg_conf_matrix = conf_matrix_sum / n_splits
    print("Average Confusion Matrix:")
    print(np.round(avg_conf_matrix).astype(int))
    plt.figure(figsize=(6, 4))
    sns.heatmap(avg_conf_matrix, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=["Predicted Normal", "Predicted Attack"],
                yticklabels=["Actual Normal", "Actual Attack"])
    plt.title("Average Confusion Matrix Across Folds")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig("average_confusion_matrix.png")
    plt.close()