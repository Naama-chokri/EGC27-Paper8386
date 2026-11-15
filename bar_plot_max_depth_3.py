import matplotlib.pyplot as plt
import numpy as np

# Dataset names
datasets = [
    'ad', 'anneal', 'audiology', 'australian-credit', 'breast-cancer', 'car',
    'drilling', 'german-credit', 'heart-cleveland', 'hepatitis', 'hypothyroid',
    'lymph', 'pima', 'primary-tumor', 'soybean', 'tic-tac-toe', 'vote', 'wdbc'
]

# Values for the second dataset provided
RF_sklearn = [
    205.03, 76.67, 81.33, 111.33, 72.36, 20.9, 54.97, 104.73, 89.9,
    67.36, 69.5, 64.06, 106.8, 31, 44.53, 27, 47.5, 80.94
]

RF_LAD_majority_vote = [
    80.77, 21.32, 21.06, 28.68, 30.56, 11.97, 21.8, 31.37, 19.74,
    15.56, 13.94, 16.76, 26.63, 14, 20.6, 16, 17.24, 18.44
]

RF_LAD_soft_vote = [
    79.86, 21.44, 20.82, 28.95, 30.9, 12.03, 21.68, 31.26, 19.95,
    15.6, 14.2, 16.74, 26.63, 14, 20.55, 16, 17.24, 18.4
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
ax.set_title('Number of features considered for each RF method by dataset when depth=3')
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
ax.set_title('Number of features considered for each RF method by dataset when depth=3', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, fontsize=20)
ax.legend(fontsize=25)

# Displaying the plot
plt.tight_layout()
plt.show()