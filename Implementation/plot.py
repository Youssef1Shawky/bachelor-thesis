import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the values from the confusion matrix


TP = 63
FN = 37
TN = 100;
FP = 0

# Create confusion matrix as a 2x2 NumPy array
conf_matrix = np.array([[TP, FN], [FP, TN]])

# Define class labels
class_names = ['Positive', 'Negative']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)  # Adjust to fit labels comfortably

# Create heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

# Add labels, title, and display the plot
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
