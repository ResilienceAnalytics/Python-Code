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
        json_filepath (str): The filepath of the JSON file containing tokenized words.
        n_components (int): The number of dimensions for t-SNE.
        random_state (int): The random state for t-SNE.
    """
    
    # Load tokenized words from JSON file
    with open(json_filepath, 'r') as file:
        tokenized_sentences = json.load(file)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Process each list of tokens
    for tokens in tokenized_sentences:
        sentence = ' '.join(tokens)
        print(f"Visualizing tokens: {tokens}")

        encoded_input = tokenizer(sentence, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        with torch.no_grad():
            outputs = model(**encoded_input)
        last_hidden_states = outputs.last_hidden_state.squeeze(0)

        # Adjust perplexity based on the length of tokens
        perplexity = min(30, len(tokens)-1)  # Ensure perplexity is less than the number of tokens

        tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        word_embeddings_2d = tsne.fit_transform(last_hidden_states.numpy())

        plt.figure(figsize=(12, 8))
        for i, token_id in enumerate(input_ids[0]):
            token = tokenizer.decode([token_id])
            plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
            plt.annotate(token, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))
        
        plt.title(f"t-SNE visualization of the tokens '{tokens}'")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.savefig(f"{json_filepath}_{tokens}_visualization.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_components", type=int, help="Number of dimensions for t-SNE")
    parser.add_argument("random_state", type=int, help="Random state for t-SNE")
    parser.add_argument("json_filepath", type=str, help="Path to the JSON file with tokenized words")
    args = parser.parse_args()

    visualize_word_embeddings_from_file(args.json_filepath, args.n_components, args.random_state)
