import sys
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_model(file_path):
    """
    Load the model from the specified file path.
    
    :param file_path: Path to the .vec file containing the embeddings.
    :return: The loaded model.
    """
    return KeyedVectors.load_word2vec_format(file_path, binary=False)

def calculate_product_difference(word, model_1, model_2):
    """
    Calculate various measures of difference for a specific word in both models.
    
    :param word: The word to compare.
    :param model_1: The first word embeddings model.
    :param model_2: The second word embeddings model.
    :return: A tuple with different measures of difference including Euclidean distance.
    """
    vector_1 = model_1[word]
    vector_2 = model_2[word]
    dn = vector_2 - vector_1

    # Element-wise multiplication: (vector2 - vector1) * vector1
    elementwise_product = dn * vector_1

    # Sum of products for each dimension: ni * dni' + dni * ni'
    sum_of_products = sum(n1 * dn_i + dn_i * n2 for n1, n2, dn_i in zip(vector_1, vector_2, dn))

    # Euclidean distance
    euclidean_distance = np.linalg.norm(dn)
    
    # Cosinus Similarity
    cosine_sim = cosine_similarity([vector_1], [vector_2])[0][0]

    return elementwise_product, sum_of_products, dn, euclidean_distance, cosine_sim

def calculate_differences_for_all_words(model_1, model_2):
    """
    Calculate various measures of differences for all common words in both models.
    
    :param model_1: The first word embeddings model.
    :param model_2: The second word embeddings model.
    :return: A dictionary with words as keys and their different measures of differences as values.
    """
    common_words = set(model_1.key_to_index.keys()).intersection(model_2.key_to_index.keys())
    differences = {word: calculate_product_difference(word, model_1, model_2) for word in common_words}
    return differences

def main(model_file_1, model_file_2, word):
    """
    Main function to load the models and calculate various measures of difference for the specified word.
    If 'ALL' is specified, calculate for all common words.
    
    :param model_file_1: The file path to the first model.
    :param model_file_2: The file path to the second model.
    :param word: The word to compare or 'ALL' for all words.
    """
    model_1 = load_model(model_file_1)
    model_2 = load_model(model_file_2)

    if word.upper() == 'ALL':
        differences = calculate_differences_for_all_words(model_1, model_2)
        for word, (elementwise_product, sum_of_products, dn, euclidean_distance, cosine_sim) in list(differences.items())[:10]:
            print(f"Word '{word}' has the following measures of differences:")
            print(f"Element-wise product: {elementwise_product}")
            print(f"Sum of products: {sum_of_products}")
            print(f"Vector of differences: {dn}")
            print(f"Euclidean distance: {euclidean_distance}")
            print(f"Cosine similarity: {cosine_sim}")
    elif word in model_1.key_to_index and word in model_2.key_to_index:
        elementwise_product, sum_of_products, dn, euclidean_distance, cosine_sim = calculate_product_difference(word, model_1, model_2)
        print(f"Word '{word}' has the following measures of differences:")
        print(f"Element-wise product: {elementwise_product}")
        print(f"Sum of products: {sum_of_products}")
        print(f"Vector of differences: {dn}")
        print(f"Euclidean distance: {euclidean_distance}")
        print(f"Cosine similarity: {cosine_sim}")
    else:
        print(f"Word '{word}' not found in one or both of the models.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_embeddings.py <model_file_1> <model_file_2> <word or 'ALL'>")
        sys.exit(1)

    model_file_1, model_file_2, word = sys.argv[1], sys.argv[2], sys.argv[3]
    main(model_file_1, model_file_2, word)

