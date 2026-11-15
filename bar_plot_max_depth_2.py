import matplotlib.pyplot as plt
import numpy as np

# Dataset names
datasets = [
    'ad', 'anneal', 'audiology', 'australian-credit', 'breast-cancer', 'car',
    'drilling', 'german-credit', 'heart-cleveland', 'hepatitis', 'hypothyroid',
    'lymph', 'pima', 'primary-tumor', 'soybean', 'tic-tac-toe', 'vote', 'wdbc']

# Values for each method
RF_sklearn = [
    131.97, 59.36, 63.26, 72.2, 55.26, 18.9, 49.7, 74.67, 60.47,
    58.1, 48.5, 54.4, 75.06, 30.63, 38.76, 26.87, 39.36, 60.24
]

RF_LAD_majority_vote = [
    51.93, 13.1, 19.44, 26.24, 28.68, 7.53, 20.68, 28.94, 18.55,
    15.37, 6.5, 16.8, 19.37, 12.06, 19.32, 16, 15.06, 17.8
]

RF_LAD_soft_vote = [
    53.14, 13.16, 19.6, 26.18, 29, 7.64, 21.06, 29.44, 18.8,
    15.53, 6.5, 16.7, 19.63, 12.3, 19.44, 15.97, 14.8, 17.45
]

# Setting positions for bars
#x = np.arange(len(datasets))
#width = 0.32  # width of bars
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
ax.set_title('Number of features considered for each RF method by dataset when depth=2', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, fontsize=20)
ax.legend(fontsize=25)

# Displaying the plot
plt.tight_layout()
plt.show()
