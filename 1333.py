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

data_filename = 'baccarat_data.csv'
num_games = 10000

try:
    training_data = pd.read_csv(data_filename)
except FileNotFoundError:
    training_data = pd.DataFrame(columns=['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2', 'outcome'])

imputer = SimpleImputer(strategy='median')
training_data[['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2']] = imputer.fit_transform(training_data[['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2']])

X = training_data[['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2']]
y = training_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
gbc = GradientBoostingClassifier(random_state=42)
mlp = MLPClassifier(random_state=42)

dtc_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 1, 3, 5, 7]}
rfc_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 1, 3, 5, 7], 'max_features': ['sqrt', 'log2']}
svc_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
gbc_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 1, 3, 5, 7], 'learning_rate': [0.01, 0.1, 1]}
mlp_params = {'hidden_layer_sizes': [(10,), (50,), (100,)], 'activation': ['relu', 'tanh', 'logistic'], 'alpha': [0.0001, 0.001, 0.01]}

dtc_grid = GridSearchCV(dtc, dtc_params, scoring='accuracy', cv=5)
rfc_random = RandomizedSearchCV(rfc, rfc_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)
svc_grid = GridSearchCV(svc, svc_params, scoring='accuracy', cv=5)
gbc_random = RandomizedSearchCV(gbc, gbc_params, n_iter=10, scoring='accuracy', cv=5, random_state=42)

models = {'Decision Tree': dtc_grid, 'Random Forest': rfc_random, 'Support Vector Machine': svc_grid, 'Gradient Boosting': gbc_random, 'Multi-layer Perceptron': mlp}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name}:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    
