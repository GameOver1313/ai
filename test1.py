import random
import numpy as np
from sklearn.linear_model import LogisticRegression

# Define the number of games to simulate
num_games = 10000

# Define the training data
training_data = []
training_labels = []
for i in range(num_games):
    # Generate a random hand for the player and banker
    player_hand = random.choices(range(1, 14), k=2)
    banker_hand = random.choices(range(1, 14), k=2)
    
    # Determine the total value of the player's hand
    player_total = sum(player_hand) % 10
    
    # Determine the total value of the banker's hand
    banker_total = sum(banker_hand) % 10
    
    # Add the game outcome and features to the training data
    if player_total > banker_total:
        training_data.append(player_hand + banker_hand)
        training_labels.append(1)
    elif player_total < banker_total:
        training_data.append(player_hand + banker_hand)
        training_labels.append(0)

# Train a logistic regression model on the training data
model = LogisticRegression().fit(training_data, training_labels)

# Use the model to predict the outcome of a new game
new_game = random.choices(range(1, 14), k=4)
prediction = model.predict([new_game])[0]

# Calculate the percentage of correct predictions
num_correct_predictions = 0
for i in range(num_games):
    prediction = model.predict([training_data[i]])[0]
    if prediction == training_labels[i]:
        num_correct_predictions += 1
percent_correct_predictions = num_correct_predictions / num_games * 100

print("The predicted outcome of the new game is: {}".format(prediction))
print("The percentage of correct predictions is: {}%".format(percent_correct_predictions))
