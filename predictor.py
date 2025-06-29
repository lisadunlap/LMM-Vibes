"""
This script aims to find the most representative properties (from clustered text data) 
for different models. It does this by training a logistic regression model to predict 
a model's identity based on the properties associated with its outputs.
L1 regularization is used to perform feature selection, identifying the most 
predictive properties for each model.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.utils import resample

file_path = 'cluster_results/arena_full_vibe_results_parsed_processed_hdbscan_clustered/arena_full_vibe_results_parsed_processed_hdbscan_clustered.csv.gz'
df = pd.read_csv(file_path)
print("Columns in the dataframe:", df.columns)
print(df.head())

# Get unique clusters and models
unique_clusters = df['property_description_fine_cluster_label'].unique()
unique_models = df['model'].unique()

cluster_to_int = {label: i for i, label in enumerate(unique_clusters)}
model_to_int = {model: i for i, model in enumerate(unique_models)}

grouped = df.groupby(['question_id', 'model'])['property_description_fine_cluster_label'].apply(list).reset_index()

X = []
y = []

for index, row in grouped.iterrows():
    feature_vector = np.zeros(len(unique_clusters), dtype=int)
    for prop in row['property_description_fine_cluster_label']:
        if prop in cluster_to_int:
            feature_vector[cluster_to_int[prop]] = 1
    X.append(feature_vector)
    y.append(model_to_int[row['model']])

X = np.array(X)
y = np.array(y)

print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Using 'saga' solver as it supports L1 penalty and multinomial loss
log_reg = LogisticRegression(penalty='l1', solver='saga', C=1.0, multi_class='multinomial', max_iter=1000, random_state=42)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Inspect coefficients
coefficients = log_reg.coef_
print(f'Shape of coefficients: {coefficients.shape}')

int_to_model = {i: model for model, i in model_to_int.items()}
int_to_cluster = {i: label for label, i in cluster_to_int.items()}

for i in range(coefficients.shape[0]):
    model_name = int_to_model[i]
    model_coeffs = coefficients[i]
    
    # Get indices of sorted coefficients in descending order
    sorted_coeffs_indices = np.argsort(model_coeffs)[::-1]
    
    print(f'\nTop 10 most predictive properties for model: {model_name}')
    count = 0
    for idx in sorted_coeffs_indices:
        # Check for non-zero coefficient
        if model_coeffs[idx] > 0 and count < 10:
            print(f'  - {int_to_cluster[idx]} (coefficient: {model_coeffs[idx]:.4f})')
            count += 1
    if count == 0:
        print('  No positive predictive properties found (all coefficients are zero or negative).')

print("\n" + "="*80)
print("Running bootstrapping for robust feature selection...")
print("="*80 + "\n")

n_bootstraps = 100
# Store non-zero coefficient counts for each feature for each model
feature_selection_counts = np.zeros_like(coefficients, dtype=int)

for i in range(n_bootstraps):
    if (i+1) % 10 == 0:
        print(f"Running bootstrap iteration {i+1}/{n_bootstraps}...")
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    log_reg_boot = LogisticRegression(penalty='l1', solver='saga', C=1.0, multi_class='multinomial', max_iter=1000, random_state=42)
    log_reg_boot.fit(X_resampled, y_resampled)
    feature_selection_counts += (log_reg_boot.coef_ != 0).astype(int)

# Report stable features
for i in range(coefficients.shape[0]):
    model_name = int_to_model[i]
    selection_counts = feature_selection_counts[i]
    
    # Get indices of sorted counts in descending order
    sorted_indices = np.argsort(selection_counts)[::-1]
    
    print(f'\nTop 10 most STABLE predictive properties for model: {model_name} (from {n_bootstraps} bootstraps)')
    count = 0
    for idx in sorted_indices:
        if selection_counts[idx] > 0 and count < 10:
            print(f'  - {int_to_cluster[idx]} (selected in {int(selection_counts[idx])}/{n_bootstraps} bootstraps)')
            count += 1
    if count == 0:
        print('  No stable predictive properties found.') 