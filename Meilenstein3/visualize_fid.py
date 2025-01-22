import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from the txt file
file_path = 'metrics_fid.txt'  # Replace with your file path
data = pd.read_csv(file_path, delim_whitespace=True)

# Set the style of the plot
sns.set(style="whitegrid")

# Create a Seaborn lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(x='Step', y='FID', data=data, marker='o')

# Add labels and title
plt.title('FID Score Over Training Steps', fontsize=16)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('FID Score', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
