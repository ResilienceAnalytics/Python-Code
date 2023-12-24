import argparse
import json
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch

def visualize_word_embeddings_from_file(json_filepath, n_components, random_state):
    """
    Visualize the construction of sentence embeddings word by word from a JSON file.
    
    Args:
        json_filepath (str): The filepath of the JSON file containing sentences to visualize.
        n_components (int): The number of dimensions for t-SNE.
        random_state (int): The random state for t-SNE.
    """
    
    # Load sentences from JSON file
    with open(json_filepath, 'r') as file:
        sentences = json.load(file)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Process each sentence
    for sentence in sentences:
        print(f"Visualizing sentence: {sentence}")
        encoded_input = tokenizer(sentence, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]

        with torch.no_grad():
            outputs = model(**encoded_input)

        last_hidden_states = outputs.last_hidden_state.squeeze(0)
        tsne = TSNE(n_components=n_components, random_state=random_state)
        word_embeddings_2d = tsne.fit_transform(last_hidden_states.numpy())

        plt.figure(figsize=(12, 8))
        for i, token in enumerate(tokens):
            plt.scatter(word_embeddings_2d[:i+1, 0], word_embeddings_2d[:i+1, 1])
            plt.annotate(token, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))
        
        plt.title(f"t-SNE visualization of the sentence '{sentence}' construction")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"{json_filepath}_visualization.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_components", type=int, help="Number of dimensions for t-SNE")
    parser.add_argument("random_state", type=int, help="Random state for t-SNE")
    parser.add_argument("json_filepath", type=str, help="Path to the JSON file with sentences")
    args = parser.parse_args()

    visualize_word_embeddings_from_file(args.json_filepath, args.n_components, args.random_state)
