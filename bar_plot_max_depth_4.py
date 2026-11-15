import matplotlib.pyplot as plt
import numpy as np

# Dataset names
datasets = [
    'ad', 'anneal', 'audiology', 'australian-credit', 'breast-cancer', 'car',
    'drilling', 'german-credit', 'heart-cleveland', 'hepatitis', 'hypothyroid',
    'lymph', 'pima', 'primary-tumor', 'soybean', 'tic-tac-toe', 'vote', 'wdbc'
]

RF_sklearn = [
    284.44, 82, 91.9, 120.5, 80.44, 21, 55.6, 109.33, 94.5,
    68, 77.1, 65.9, 119.53, 31, 45.9, 27, 47.97, 87
]

RF_LAD_majority_vote = [
    91.83, 23.4, 21.26, 28.95, 31.4, 14.4, 21.9, 32, 20.05,
    15.56, 23.74, 16.87, 26.68, 14, 20.7, 16, 18.32, 18.94
]

RF_LAD_soft_vote = [
    91.67, 23.4, 21.55, 28.94, 31.4, 14.63, 22.24, 31.76, 19.94,
    15.53, 23.74, 16.8, 26.68, 14, 20.7, 16, 18.63, 18.56
]

"""
# Setting positions for bars
x = np.arange(len(datasets))
width = 0.25  # width of bars

# Creating the figure and the bars
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - width, RF_sklearn, width, label='RF-sklearn', color='blue')
ax.bar(x, RF_LAD_majority_vote, width, label='RF_LAD_majority-vote', color='green')
ax.bar(x + width, RF_LAD_soft_vote, width, label='RF_LAD_soft-vote', color='red')

# Adding labels and title
ax.set_xlabel('Datasets')
ax.set_ylabel('Number of features')
ax.set_title('Number of features considered for each RF method by dataset when depth=4')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45)
ax.legend()

# Displaying the plot
plt.tight_layout()
plt.show()
"""
# Adjusting the spacing between each dataset by increasing space between positions
width = 1
spacing_factor = 5  # Adjust this factor to increase or decrease spacing
x = np.arange(len(datasets)) * spacing_factor

# Creating the figure and the bars
fig, ax = plt.subplots(figsize=(20, 10))

ax.bar(x - width, RF_sklearn, width, label='RF-sklearn', color='blue')
ax.bar(x, RF_LAD_majority_vote, width, label='RF-LAD-majority-vote', color='green')
ax.bar(x + width, RF_LAD_soft_vote, width, label='RF-LAD-soft-vote', color='red')

# Adding labels and title
ax.set_xlabel('Datasets', fontsize=25)
ax.set_ylabel('Number of features', fontsize=25)
ax.set_title('Number of features considered for each RF method by dataset when depth=4', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, fontsize=20)
ax.legend(fontsize=25)

# Displaying the plot
plt.tight_layout()
plt.show()