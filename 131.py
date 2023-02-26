import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Define the filename for the training data
data_filename = 'baccarat_data.csv'

# Define the number of games to simulate
num_games = 10000

# Load the existing training data, or create a new empty dataframe
try:
    training_data = pd.read_csv(data_filename, index_col=0)
except FileNotFoundError:
    training_data = pd.DataFrame(columns=['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2', 'outcome'])

# Fill missing values in the training data with median values
imputer = SimpleImputer(strategy='median')
training_data = imputer.fit_transform(training_data)

# Split the training data into input features (X) and output variable (y)
X = training_data[:, :-1]
y = training_data[:, -1]

# Normalize and scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)
mlp = MLPClassifier(random_state=42)

# Define the hyperparameters to tune for each model using GridSearchCV or RandomizedSearchCV
dtc_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 1, 3, 5, 7]}
rfc_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 1, 3, 5, 7], 'max_features': ['sqrt', 'log2']}
svc_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
gbc_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 1, 3, 5, 7], 'learning_rate': [0.01, 0.1, 1]}
mlp_params = {'hidden_layer_sizes': [(10,), (50,), (100,)], 'activation': ['relu', 'tanh', 'logistic'], 'alpha': [0.0001, 0.001, 0.01]}

# Use GridSearchCV or RandomizedSearchCV to find the best hyperparameters for each model
dtc_grid = GridSearchCV(dtc, dtc_params, scoring='accuracy', cv=5)
rfc_random = RandomizedSearchCV(rfc, rfc_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)
svc_grid = GridSearchCV(svc, svc_params, scoring='accuracy', cv=5)
gbc_random = RandomizedSearchCV(gbc, gbc_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)

