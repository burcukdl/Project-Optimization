import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define file path templates for each optimizer
file_templates = {
    "Gradient Descent": os.path.join(base_dir, "../results/w1_trajectory_gd_{}.csv"),
    "SGD": os.path.join(base_dir, "../results/w1_trajectory_sgd_{}.csv"),
    "Adam": os.path.join(base_dir, "../results/w1_trajectory_adam_{}.csv")
}

# t-SNE configuration
tsne_params = {
    "n_components": 2,
    "random_state": 42,
    "perplexity": 70,
    "max_iter": 300
}

# Create a figure 
plt.figure(figsize=(18, 5))

for i, (name, template) in enumerate(file_templates.items(), 1):
    all_tsne_results = []
    all_labels = []
    
    # Read data from multiple initializations (w1 to w5)
    for j in range(1, 6):
        file_path = template.format(j)
        data = pd.read_csv(file_path)
        
        # Select weight columns (excluding Epoch)
        weights = data.iloc[:, 1:]
        
        # Apply t-SNE
        tsne = TSNE(**tsne_params)
        tsne_results = tsne.fit_transform(weights)
        
        all_tsne_results.append(tsne_results)
        all_labels.extend([f"start{j}"] * len(tsne_results))
    
    # Combine results into a single DataFrame
    tsne_combined = pd.DataFrame(
        data=np.vstack(all_tsne_results),
        columns=['Dimension 1', 'Dimension 2']
    )
    tsne_combined['Start'] = all_labels
    
    # Plot t-SNE results
    plt.subplot(1, 3, i)
    scatter = plt.scatter(
        tsne_combined['Dimension 1'],
        tsne_combined['Dimension 2'],
        c=tsne_combined['Start'].astype('category').cat.codes,
        cmap='tab10',
        s=50,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Start')
    plt.title(f"t-SNE for {name}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "../results/2all_tsne_results.png"))
plt.show()
