import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys

def visualize_embeddings(embeddings_path, n_components, random_state, output_dir):
    """
    Visualize BERT embeddings using t-SNE and save the plot to a specified directory.

    Args:
    embeddings_path (str): Path to the .npy file containing BERT embeddings.
    n_components (int): Number of dimensions for t-SNE.
    random_state (int): Random state for reproducibility in t-SNE.
    output_dir (str): Directory where the plot image will be saved.

    This function loads BERT embeddings from a .npy file, applies t-SNE for dimensionality 
    reduction, and saves the resulting plot as an image in the specified directory.
    """
    # Load embeddings from the .npy file
    embeddings = np.load(embeddings_path)

    # Create a t-SNE model with the given parameters
    tsne = TSNE(n_components=n_components, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate and save the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], edgecolors='k', c='orange')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('2D Visualization of BERT Embeddings using t-SNE')
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python visualize_embeddings.py <path_to_npy_file> <n_components> <random_state> <output_directory>")
        sys.exit(1)

    embeddings_file = sys.argv[1]
    n_comp = int(sys.argv[2])
    rand_state = int(sys.argv[3])
    out_dir = sys.argv[4]

    visualize_embeddings(embeddings_file, n_comp, rand_state, out_dir)
