import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Define the filename for the training data
data_filename = 'baccarat_data.csv'

# Define the number of games to simulate
num_games = 10000

# Load the existing training data, or create a new empty dataframe
try:
    training_data = pd.read_csv(data_filename, index_col=0)
except FileNotFoundError:
    training_data = pd.DataFrame(columns=['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2', 'outcome'])

# Define the number of decks being used
num_decks = 8

# Generate the deck of cards
cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0] * 4 * num_decks

# Initialize the models
models = [DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier()]

# Train and evaluate each model
for model in models:
    # Split the data into training and testing sets
    X = training_data[['player_hand_1', 'player_hand_2', 'banker_hand_1', 'banker_hand_2']].values
    y = training_data['outcome'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    accuracy = model.score(X_test, y_test)
    print(f"{model.__class__.__name__} accuracy: {accuracy:.2%}")

# Start the game loop
while True:
    # Prompt the user for the outcome of the previous game
    previous_outcome = input("What was the outcome of the previous game? Enter 'player', 'banker', or 'tie', or 'stop' to quit: ")
    if previous_outcome == 'stop':
        break

    # Generate the player's and banker's hands
    player_hand = [random.choice(cards), random.choice(cards)]
    banker_hand = [random.choice(cards), random.choice(cards)]

    # Predict the outcome using each model and print the probabilities
    for model in models:
        probas = model.predict_proba([[player_hand[0], player_hand[1], banker_hand[0], banker_hand[1]]])[0]
        print(f"{model.__class__.__name__} probabilities: player={probas[1]:.2%}, banker={probas[0]:.2%}, tie={probas[2]:.2%}")

    # Prompt the user for the actual outcome and add the data to the training set
    if previous_outcome == 'player':
        training_data = training_data.append({'player_hand_1': player_hand[0], 'player_hand_2': player_hand[1],
                                              'banker_hand_1': banker_hand[0], 'banker_hand_2': banker_hand[1],
                                              'outcome': 1}, ignore_index=True)
    elif previous_outcome == 'banker':
        training_data = training_data.append({'player_hand_1': player_hand[0], 'player_hand_2': player_hand[1],
                                              'banker_hand_1': banker_hand[0], 'banker_hand_2': banker_hand[1],
                                              'outcome': 0}, ignore
