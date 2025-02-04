# Implement random forest to classify spam twitter accounts
# Twitter data in the data folder

import pandas as pd
import numpy as np
from collections import Counter
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os

# start by taking 75% of the labeled data -- our training dataset
# then seperate the rest 25% of data -- our testing dataset
# Load the features file
features_df = pd.read_csv('./Data/tweet_features.csv')

# Load the labels file
labeled_df = pd.read_csv('./Data/labeled_tweets.csv')

# select features for decision tree
selected_features = ['user_age', 'user_favs', 'hashtag_count', 'text_length', 'follower_count', 'friends_count', 'status_count', 'listed_count']
x = features_df[selected_features].values

# select labels
y = labeled_df['spam_label'].values

# Train-test split data 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=35)

# helper functions to aid splitting
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

# added gini aswell
def gini_impurity(y):
    proportions = np.bincount(y) / len(y)
    return 1 - np.sum(proportions**2)

def information_gain(y, y_left, y_right):
    p = len(y_left) / len(y)
    return gini_impurity(y) - p * gini_impurity(y_left) - (1 - p) * gini_impurity(y_right)

def split(X, y, feature_index, threshold):
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

# Decision tree implementation
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth == self.max_depth:
            self.tree = np.bincount(y).argmax()  # Return majority class
            return
        
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                _, _, y_left, y_right = split(X, y, feature_idx, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        if best_gain == -1:
            self.tree = np.bincount(y).argmax()
            return

        left_indices = X[:, split_idx] <= split_threshold
        right_indices = X[:, split_idx] > split_threshold
        self.tree = {
            'feature_index': split_idx,
            'threshold': split_threshold,
            'left': DecisionTree(self.max_depth, self.min_samples_split),
            'right': DecisionTree(self.max_depth, self.min_samples_split)
        }
        self.tree['left'].fit(X[left_indices], y[left_indices], depth + 1)
        self.tree['right'].fit(X[right_indices], y[right_indices], depth + 1)

    def predict_single(self, x):
        if not isinstance(self.tree, dict):
            return self.tree
        if x[self.tree['feature_index']] <= self.tree['threshold']:
            return self.tree['left'].predict_single(x)
        return self.tree['right'].predict_single(x)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

# Random forest implementation
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)

# fitting
rf = RandomForest(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)

# prediction
y_pred = rf.predict(X_test)

# sklearn performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
