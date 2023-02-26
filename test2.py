import random
import numpy as np
from sklearn.linear_model import LogisticRegression
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

# Initialize the previous outcome variable to None
previous_outcome = None

# Loop until the user enters 'stop'
while True:
    # Prompt the user for the outcome of the previous game
    if previous_outcome is None:
        previous_outcome = input("What was the outcome of the previous game? Enter 'player', 'banker', or 'tie', or 'stop' to exit: ")
    else:
        previous_outcome = input("What was the outcome of the previous game? Enter 'player', 'banker', or 'tie', or 'stop' to exit: ")
        
    # Exit the loop if the user enters 'stop'
    if previous_outcome == 'stop':
        break

    # Add the previous game to the training data
    if previous_outcome == 'player':
        player_hand = [int(card) for card in input("Enter the player's hand (e.g. '4 7'): ").split()]
        banker_hand = [int(card) for card in input("Enter the banker's hand (e.g. '2 3'): ").split()]
        training_data = training_data.append({'player_hand_1': player_hand[0], 'player_hand_2': player_hand[1],
                                              'banker_hand_1': banker_hand[0], 'banker_hand_2': banker_hand[1],
                                              'outcome': 1}, ignore_index=True)
    elif previous_outcome == 'banker':
        player_hand = [int(card) for card in input("Enter the player's hand (e.g. '4 7'): ").split()]
        banker_hand = [int(card) for card in input("Enter the banker's hand (e.g. '2 3'): ").split()]
        training_data = training_data.append({'player_hand_1': player_hand[0], 'player_hand_2': player_hand[1],
                                              'banker_hand_1': banker_hand[0], 'banker_hand_2': banker_hand[1],
                                              'outcome': 0}, ignore_index=True)
    elif previous_outcome == 'tie':
        player_hand = [int(card) for card in input("Enter the player's hand (e.g. '4 7'): ").split()]
        banker_hand = [int(card) for card in input("Enter the banker's hand (e.g. '2 3'): ").split()]
        training_data = training_data.append({'player_hand_1': player_hand[0], 'player_hand_2': player_hand[1],
                                              'banker_hand_1': banker_hand[0], 'banker_hand_2': banker_hand[1],
                                              'outcome': 2}, ignore_index=True)
    else:
        print("Invalid input. The previous outcome will not be added to the database.")

    # Save the training data to a file
    training_data.to_csv(data_filename)

    #

