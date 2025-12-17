#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:50:31 2024

@author: yizhen
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from dmba import classificationSummary
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score  
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import learning_curve

df =pd.read_csv('/Users/yizhen/Desktop/6105hw/assignment4/accidents-Assignment4.csv')
df.shape
df.head()
df.dtypes
summary_stats = df.describe()
mode_data = df.mode().iloc[0]
summary_stats.loc['mode'] = mode_data

print(df.isnull().sum())
#no missing values as check

import matplotlib.pyplot as plt
import seaborn as sns

outlier_cols = ['SPD_LIM']
for col in outlier_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# categorical_columns = df.select_dtypes(include=['object']).columns
# df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
# df_encoded.head()

from sklearn.preprocessing import StandardScaler
X = df.drop('MAX_SEV', axis=1)
y = df['MAX_SEV']
y = pd.get_dummies(y, drop_first=True, dtype=int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)


print(feature_importance)
#remove speed limit

from sklearn.model_selection import train_test_split
X = df.drop(['WEATHER_adverse', 'INT_HWY', 'WRK_ZONE', 'MAX_SEV'], axis=1)
y = df['MAX_SEV']

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
le = LabelEncoder()
y = le.fit_transform(df['MAX_SEV'])
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

X_train, X_temp, y_train, Y_temp = train_test_split(X, y, test_size=0.30, random_state=1)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=1)
print("Training set size:", X_train.shape)
print("Validation set size:", X_validation.shape)
print("Test set size:", X_test.shape)

#part2
architectures = [
    (4,),
    (4,4),
    (8, 8),
    (16, 16),
    (16, 8, 4),
    (32, 16, 8),
    (32,32,32),
    (64,64,32),
    (64,64,64,16),
    (64,64,64,64)
]
results_relu = []
results_tanh = []


for hidden_layers in architectures:
    # ReLu
    clf_relu = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=1
    )
    

    clf_tanh = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='tanh',
        solver='adam',
        max_iter=1000,
        random_state=1
    )


    clf_relu.fit(X_train, y_train)
    clf_tanh.fit(X_train, y_train)

    train_acc_relu = accuracy_score(y_train, clf_relu.predict(X_train))
    val_acc_relu = accuracy_score(y_validation, clf_relu.predict(X_validation))
    val_loss_relu = log_loss(y_validation, clf_relu.predict_proba(X_validation))
    

    train_acc_tanh = accuracy_score(y_train, clf_tanh.predict(X_train))
    val_acc_tanh = accuracy_score(y_validation, clf_tanh.predict(X_validation))
    val_loss_tanh = log_loss(y_validation, clf_tanh.predict_proba(X_validation))
    
    results_relu.append({
        'architecture': hidden_layers,
        'train_accuracy': train_acc_relu,
        'validation_accuracy': val_acc_relu,
        'validation_loss': val_loss_relu,
        'activation': 'ReLU'
    })
    
    results_tanh.append({
        'architecture': hidden_layers,
        'train_accuracy': train_acc_tanh,
        'validation_accuracy': val_acc_tanh,
        'validation_loss': val_loss_tanh,
        'activation': 'Tanh'
    })

results_df = pd.DataFrame(results_relu + results_tanh)
print(results_df)





best_model_relu = MLPClassifier(
    hidden_layer_sizes=(32,16,8), 
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=1,
    early_stopping=True,  
    validation_fraction=0.1,  
    learning_rate_init=0.02,
    n_iter_no_change=20  
)

best_model_tanh = MLPClassifier(
    hidden_layer_sizes=(32,16,8), 
    activation='tanh',
    solver='adam',
    max_iter=1000,
    random_state=1,
    early_stopping=True,  
    validation_fraction=0.1,  
    learning_rate_init=0.02,
    n_iter_no_change=20  
)

print("\nReLU Model Training:")
best_model_relu.fit(X_train, y_train)
print(f"Number of iterations: {best_model_relu.n_iter_}")
print(f"Training set accuracy: {best_model_relu.score(X_train, y_train):.4f}")
print(f"Validation set accuracy: {best_model_relu.score(X_validation, y_validation):.4f}")
print(f"Test set accuracy: {best_model_relu.score(X_test, y_test):.4f}")

# Train Tanh model and print information
print("\nTanh Model Training:")
best_model_tanh.fit(X_train, y_train)
print(f"Number of iterations: {best_model_tanh.n_iter_}")
print(f"Training set accuracy: {best_model_tanh.score(X_train, y_train):.4f}")
print(f"Validation set accuracy: {best_model_tanh.score(X_validation, y_validation):.4f}")
print(f"Test set accuracy: {best_model_tanh.score(X_test, y_test):.4f}")

tanh_accuracies = []
for i in range(1,10):
    best_model_tanh = MLPClassifier(
        hidden_layer_sizes=(32,16,8), 
        activation='tanh',
        solver='adam',
        max_iter=1000,
        random_state=i,
        early_stopping=True,  
        validation_fraction=0.1,  
        learning_rate_init=0.02,
        n_iter_no_change=20 
    )
    
    best_model_tanh.fit(X_train, y_train)
    accuracy = best_model_tanh.score(X_test, y_test)
    tanh_accuracies.append(accuracy)

mean_accuracy = np.mean(tanh_accuracies)
std_accuracy = np.std(tanh_accuracies)

print("random state metrics:")
print(f"average accuracy: {mean_accuracy:.4f}")
print(f"standard deviation: {std_accuracy:.4f}")


y_val_pred = best_model_tanh.predict(X_validation)
y_test_pred = best_model_tanh.predict(X_test)




f1_val = f1_score(y_validation, y_val_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')
print("F1score:")
print(f"validation F1: {f1_val:.4f}")
print(f"Test F1: {f1_test:.4f}")


train_sizes, train_scores, val_scores = learning_curve(
    best_model_tanh, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, n_jobs=-1, scoring='accuracy'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='validation score')
plt.xlabel('training samples')
plt.ylabel('accuracy')
plt.title('learning rate')
plt.legend(loc='best')
plt.grid(True)
plt.show()


train_accuracy = best_model_tanh.score(X_train, y_train)
test_accuracy = best_model_tanh.score(X_test, y_test)
print("overfit check:")
print(f"training accuracy: {train_accuracy:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
print(f"difference: {train_accuracy - test_accuracy:.4f}")

# Calculate and print detailed evaluation metrics
def print_diagnostic_metrics(y_true, y_pred, dataset_name):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    
    # Calculate precision, sensitivity, and specificity for each class
    n_classes = len(np.unique(y_true))
    metrics = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics.append({
            'Class': i,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })
    
    print(f"\n{dataset_name} Diagnostic Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f}")
    print("\nDetailed metrics for each class:")
    for m in metrics:
        print(f"\nClass {m['Class']}:")
        print(f"Precision: {m['Precision']:.4f}")
        print(f"Sensitivity (Recall): {m['Sensitivity']:.4f}")
        print(f"Specificity: {m['Specificity']:.4f}")

# Function to plot confusion matrices
def plot_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Validation set confusion matrix
    cm_val = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Validation Set Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Test set confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Test Set Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()

# Perform diagnostic checks
print_diagnostic_metrics(y_validation, y_val_pred, "Validation Set")
print_diagnostic_metrics(y_test, y_test_pred, "Test Set")

# Plot confusion matrices
plot_confusion_matrices(y_validation, y_val_pred, y_test, y_test_pred)

def simulate_variable_impact(model, X, feature_name, num_points=100):
    # Create range of values for the feature
    feature_idx = X.columns.get_loc(feature_name)
    feature_min = X[feature_name].min()
    feature_max = X[feature_name].max()
    x_range = np.linspace(feature_min, feature_max, num_points)
    
    # Create predictions for each value
    predictions = []
    for x_val in x_range:
        X_temp = X.iloc[0:1].copy()  # Use first row as template
        X_temp.iloc[0, feature_idx] = x_val
        pred = model.predict_proba(X_temp)[0]
        predictions.append(pred)
    
    # Convert to numpy array for easier plotting
    predictions = np.array(predictions)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(predictions.shape[1]):
        plt.plot(x_range, predictions[:, i], label=f'Class {i}')
    
    plt.xlabel(feature_name)
    plt.ylabel('Probability')
    plt.title(f'Impact of {feature_name} on Class Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()


for feature in X.columns:
    simulate_variable_impact(best_model_tanh, X, feature)







