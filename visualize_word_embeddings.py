import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def visualize_semantic_development(sentence, n_components, random_state):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    tokenized_sentence = tokenizer.tokenize(sentence)
    cumulative_embeddings = []

    for i in range(len(tokenized_sentence)):
        partial_sentence = ' '.join(tokenized_sentence[:i+1])
        encoded_input = tokenizer(partial_sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded_input)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        cumulative_embeddings.extend(embeddings[-1:])

        if len(cumulative_embeddings) > 1:
            cumulative_embeddings_np = np.array(cumulative_embeddings)

            # Ajustez la perplexité en fonction du nombre d'embeddings
            perplexity = min(len(cumulative_embeddings) - 1, 30)
            
            tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(cumulative_embeddings_np)

            plt.figure(figsize=(12, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
            for j, word in enumerate(tokenized_sentence[:i+1]):
                plt.annotate(word, (embeddings_2d[j, 0], embeddings_2d[j, 1]))
            plt.title(f"Semantic development up to word: '{word}'")
            plt.xlabel("t-SNE feature 1")
            plt.ylabel("t-SNE feature 2")
            plt.savefig(f"semantic_development_{i}.png")
            plt.close()

# Exemple d'utilisation
sentence = "The quick brown fox jumps over the lazy dog"
visualize_semantic_development(sentence, 2, 42)
