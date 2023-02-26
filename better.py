import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Define the filename for the training data
data_filename = 'baccarat_data.csv'

# Define the number of games to simulate
num_games = 10000

# Load the existing training data, or create a new empty dataframe
try:
    training_data = pd.read_csv(data_filename, index_col=0)
except FileNotFoundError:
    training_data = pd.DataFrame(columns=['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2', 'outcome'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data[['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2']], training_data['outcome'], test_size=0.3, random_state=42)

# Initialize the models
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
nn_model = MLPClassifier()

# Train the models on the training data
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
nn_model.fit(X_train, y_train)

# Evaluate the performance of the models on the testing data
dt_score = dt_model.score(X_test, y_test)
rf_score = rf_model.score(X_test, y_test)
nn_score = nn_model.score(X_test, y_test)

# Determine the model with the highest accuracy
if dt_score >= rf_score and dt_score >= nn_score:
    best_model = dt_model
elif rf_score >= dt_score and rf_score >= nn_score:
    best_model = rf_model
else:
    best_model = nn_model

# Start the game loop
while True:
    # Prompt the user for the outcome of the previous game
    previous_outcome = input("What was the outcome of the previous game? Enter 'player', 'banker', or 'tie', or 'stop' to quit: ")
    if previous_outcome == 'stop':
        break

    # Add the previous game to the training data
    if previous_outcome == 'player':
        player_hand = [int(card) for card in input("Enter the player's hand (e.g. '4 7'): ").split()]
        banker_hand = [int(card) for card in input("Enter the banker's hand (e.g. '2 3'): ").split()]
        training_data = training_data.append({'player_hand_1': player_hand[0], 'player_hand_2': player_hand[1],
                                              'banker_hand_1': banker_hand[0], 'banker_hand_2': banker_hand[1],
                                              'outcome': 
