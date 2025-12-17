#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:05:05 2024

@author: yizhen
"""
#part1
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv('/Users/yizhen/Desktop/6105hw/assignment3/accidents-Assignment_3.csv')
df.shape
df.head()
df.dtypes
#descriptive value + matrix
summary_stats = df.describe()
mode_data = df.mode().iloc[0]
summary_stats.loc['mode'] = mode_data

#Part2
print(df.isnull().sum())
# use mode value for na value
for column in df.columns:
    if df[column].isna().sum() > 0:
        if column == 'SPD_LIM':
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
        else:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

outlier_cols = ['SPD_LIM']
for col in outlier_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
    
#No Outlier
X = df.drop(columns='MAX_SEV')
Y = df['MAX_SEV']
Y = pd.get_dummies(Y, drop_first=True, dtype=int)
#Y_train = Y_train.ravel()



# Y_dummies = pd.get_dummies(df['MAX_SEV'], dtype=int)
# Y = Y_dummies['injury']

from sklearn.model_selection import train_test_split


X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.30, random_state=1)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=1)
print("Training set size:", X_train.shape)
print("Validation set size:", X_validation.shape)
print("Test set size:", X_test.shape)

#Part3
#logisticregression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')

logit_reg.fit(X_train, Y_train)
print('intercept ', logit_reg.intercept_[0])
print(pd.DataFrame({'coeff': logit_reg.coef_[0]},  index=X.columns).transpose())

coeff = pd.DataFrame({'coeff': logit_reg.coef_[0]},  index=X.columns).transpose()

import statsmodels.api as sm

# Add a constant to the features for the intercept term
X_train_sm = sm.add_constant(X_train)

# Fit the model using statsmodels to get p-values
logit_model = sm.Logit(Y_train, X_train_sm)
result = logit_model.fit()

# Print the summary which includes p-values
print(result.summary())


logit_reg_pred = logit_reg.predict(X_validation)
logit_reg_proba = logit_reg.predict_proba(X_validation)
cm = confusion_matrix(Y_validation, logit_reg_pred)

# Compute accuracy
accuracy = accuracy_score(Y_validation, logit_reg_pred)


# Calculate percentage of misclassifications
misclassification_rate = 1 - accuracy

# Display results
print("Confusion Matrix:")
print(cm)
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Percentage of Misclassifications: {misclassification_rate * 100:.2f}%")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier,  DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

param_grid = {
  'max_depth': [10, 20, 30, 40],
  'min_samples_split': [20, 40, 60, 80, 100],
  'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01],
}


# Which values are best?
# n_jobs=-1 will utilize all available CPUs
#cv means cross validation
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), 
   param_grid, cv=5, n_jobs=-1) 
gridSearch.fit(X_train, Y_train)
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)

#to score base on f1
# gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), 
#                           param_grid, 
#                           cv=5, 
#                           scoring='f1', 
#                           n_jobs=-1)
# gridSearch.fit(X_train, Y_train)

# print('Best score: ', gridSearch.best_score_)
# print('Best parameters: ', gridSearch.best_params_)

#decision tree
param_grid = {
'max_depth': list(range(2, 16)), # 14 values
'min_samples_split': list(range(35, 48)), 
'min_impurity_decrease': [0, 0.001, 0.0011], # 3 values
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), 
    param_grid, cv=5, n_jobs=-1)
gridSearch.fit(X_train, Y_train)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)
bestClassTree = gridSearch.best_estimator_

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
tree.plot_tree(bestClassTree, 
               feature_names=X_train.columns, 
               class_names=['No Injury', 'Injury'], 
               filled=True)
plt.show()


#accuracy:0.55

#random forest
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500], 
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}


grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=1), param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, Y_train)

print('Random Forest Initial Score:', grid_search_rf.best_score_)
print('Random Forest Initial Parameters:', grid_search_rf.best_params_)


param_grid_rf = {
    'n_estimators': [100, 130, 160, 180], 
    'max_depth': list(range(8, 15)),
    'min_samples_split': list(range(8, 14)),
}


grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=1), param_grid_rf, cv=5, n_jobs=-1)
grid_search_rf.fit(X_train, Y_train)

print('Random Forest Best Score:', grid_search_rf.best_score_)
print('Random Forest Best Parameters:', grid_search_rf.best_params_)

# Random Forest Best Score: 0.5285714285714286
# Random Forest Best Parameters: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_split': 10, 'n_estimators': 100}
best_rf = grid_search_rf.best_estimator_

# Get feature importances from the best model
importances = best_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf.estimators_], axis=0)

# Create a DataFrame to display feature importances
df1 = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances,
    'std': std
})

# Sort the DataFrame by importance
df1 = df1.sort_values('importance')

# Plot feature importances with error bars
ax = df1.plot(kind='barh', xerr='std', x='feature', legend=False, figsize=(10, 8))
ax.set_ylabel('')  # Remove y-axis label for clarity
ax.set_xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.show()

#KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(X_train)
valid_X_scaled = scaler.transform(X_validation)
test_X_scaled = scaler.transform(X_test)

results = []
for k in range(1, 15):
   knn = KNeighborsClassifier(n_neighbors=k).fit(train_X_scaled,Y_train)
   results.append({
       'k': k,
       'accuracy': accuracy_score(Y_validation, 
             knn.predict(valid_X_scaled))
})
print(results)
#k = 7, with highest score 0.6111

k_values = [result['k'] for result in results]
accuracies = [result['accuracy'] for result in results]

# Plotting the accuracy vs k
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Validation Accuracy')
plt.xticks(k_values)  # Ensures each k value is labeled on the x-axis
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs. Number of Neighbors (k)')
plt.legend()
plt.grid(True)
plt.show()

knn_best=

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def evaluate_model(model, X, Y, dataset_name):
    Y_pred = model.predict(X)
    cm = confusion_matrix(Y, Y_pred)
    accuracy = accuracy_score(Y, Y_pred)
    precision = precision_score(Y, Y_pred)
    recall = recall_score(Y, Y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    
    print(f"=== {dataset_name} Metrics ===")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print("\n")
    
# Accuracy: (TP + TN) / (TP + TN + FP + FN)
# Error Rate: 1 - Accuracy
# Precision: TP / (TP + FP)
# Sensitivity (Recall): TP / (TP + FN)
# Specificity: TN / (TN + FP)


print("Logistic Regression:")
evaluate_model(logit_reg, X_validation, Y_validation, "Validation")

print("Decision Tree:")
evaluate_model(gridSearch.best_estimator_, X_validation, Y_validation, "Validation")

print("Random Forest:")
evaluate_model(grid_search_rf.best_estimator_, X_validation, Y_validation, "Validation")

print("KNN:")
knn_best = KNeighborsClassifier(n_neighbors=7).fit(train_X_scaled, Y_train)
evaluate_model(knn_best, valid_X_scaled, Y_validation, "Validation")


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
print(vif_data)





import numpy as np

feature_names = X_train.columns  


def simulate_feature_effect_classification(model, X_train, feature_name, X_mean, feature_min, feature_max, scaler=None):

    X_sim_min = X_mean.copy()
    X_sim_max = X_mean.copy()


    X_sim_min[feature_name] = feature_min
    X_sim_max[feature_name] = feature_max


    if scaler:
        X_sim_min = scaler.transform([X_sim_min])
        X_sim_max = scaler.transform([X_sim_max])
    else:

        X_sim_min = np.array(X_sim_min).reshape(1, -1)
        X_sim_max = np.array(X_sim_max).reshape(1, -1)


    prob_min = model.predict_proba(X_sim_min)[0][1]  
    prob_max = model.predict_proba(X_sim_max)[0][1]

    return prob_min, prob_max


X_mean_class = X_train.mean()


for feature_name in X_train.columns:
    feature_min = X_train[feature_name].min()
    feature_max = X_train[feature_name].max()


    #log_prob_min, log_prob_max = simulate_feature_effect_classification(logit_reg, X_train, feature_name, X_mean_class, feature_min, feature_max)
    #print(f"Logistic Regression: {feature_name} min={feature_min} -> prob={log_prob_min:.2f}, max={feature_max} -> prob={log_prob_max:.2f}")


    tree_prob_min, tree_prob_max = simulate_feature_effect_classification(bestClassTree, X_train, feature_name, X_mean_class, feature_min, feature_max)
    print(f"Decision Tree: {feature_name} min={feature_min} -> prob={tree_prob_min:.2f}, max={feature_max} -> prob={tree_prob_max:.2f}")


    rf_prob_min, rf_prob_max = simulate_feature_effect_classification(best_rf, X_train, feature_name, X_mean_class, feature_min, feature_max)
    print(f"Random Forest: {feature_name} min={feature_min} -> prob={rf_prob_min:.2f}, max={feature_max} -> prob={rf_prob_max:.2f}")


    knn_prob_min, knn_prob_max = simulate_feature_effect_classification(knn_best, train_X_scaled, feature_name, X_mean_class, feature_min, feature_max, scaler=scaler)
    print(f"KNN: {feature_name} min={feature_min} -> prob={knn_prob_min:.2f}, max={feature_max} -> prob={knn_prob_max:.2f}")
