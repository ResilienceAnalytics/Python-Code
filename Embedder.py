import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model 
from PIL import Image
import io
import re
import sys
import torch
import numpy as np

# Tokenizers initialization 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

# Define a function to normalize text
def normalize_text(text):
    """
    Normalize text by converting to lowercase and removing non-standard punctuation.
    
    Args:
        text (str): Text to be normalized.

    Returns:
        str: Normalized text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text


    def classify_term(self, term):
        """
        Classify a given term according to its domain.

        Args:
            term (str): The term to classify.

        Returns:
            str: The domain of the term.
        """
        return self.dataset.get(term, "Unknown")

def extract_text_and_images_from_pdf(pdf_path):
    """
    Extracts text and images from a PDF file.
    
    Args:
    pdf_path (str): The file path of the PDF to be processed.

    Returns:
    tuple: Returns two items; a string containing all the text extracted from the PDF, 
           and a list of images found in the PDF.
    """
    text = ""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            images.extend(page.images)
    return text, images

def apply_ocr_to_images(images):
    """
    Applies Optical Character Recognition (OCR) to a list of images.

    Args:
    images (list): A list of images to apply OCR on.

    Returns:
    list: A list of strings where each string is the OCR result of an image.
    """
    ocr_results = []
    for img in images:
        im = Image.open(io.BytesIO(img['stream']))
        ocr_results.append(pytesseract.image_to_string(im))
    return ocr_results

def tokenize_text(text, tokenization_method, remove_stopwords=False, use_stemming=False, use_lemmatization=False):
    """
    Tokenize the given text based on the specified method.

    Args:
        text (str): Text to tokenize.
        method (str): Method of tokenization ('word', 'subword', 'letter').
        remove_stopwords (bool): Whether to remove stop words.
        use_stemming (bool): Whether to apply stemming.
        use_lemmatization (bool): Whether to apply lemmatization.

    Returns:
        list: A list of tokenized content.
    """
    text = normalize_text(text)  # Normalize the text first
    lines = text.split('\n')  # Split the text into lines
    
    tokenized_content = []
    for line in lines:
        if tokenization_method == 'word':
            tokens = word_tokenize(line)
        elif tokenization_method == 'subword':
            tokens = bert_tokenizer.tokenize(line)
        elif tokenization_method == 'letter':
            tokens = list(line)
        else:
            raise ValueError("Invalid tokenization method. Choose 'word', 'subword', or 'letter'.")

        if remove_stopwords:
            tokens = [token for token in tokens if token not in stop_words]
        if use_stemming:
            tokens = [porter_stemmer.stem(token) for token in tokens]
        if use_lemmatization:
            tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]

        tokenized_content.append(tokens)

    return tokenized_content

def generate_embeddings(tokenized_sentences, embedding_method):
    """
    Generate embeddings for the given tokenized sentences based on the specified method.

    Args:
        tokenized_sentences (list of list of str): Tokenized sentences.
        method (str): Embedding method ('word2vec', 'bert', 'gpt', 'glove').

    Returns:
        Object: A trained model or embeddings.
    """
    if embedding_method == "word2vec":
        print("Using word2vec for embeddings")
        model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
        return model

    elif embedding_method == "bert":
        print("Using bert for embeddings")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = []
        for sentence in tokenized_sentences:
            inputs = tokenizer(' '.join(sentence), return_tensors="pt")
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
        return np.vstack(embeddings)

    elif embedding_method == "gpt":
        print("Using GPT-2 for embeddings")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        embeddings = []
        for sentence in tokenized_sentences:
            if not sentence:
                continue
            inputs = tokenizer(' '.join(sentence), return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
        return np.vstack(embeddings)

    elif embedding_method == "glove":
        print("Using glove for embeddings")
        glove_model = KeyedVectors.load_word2vec_format('path/to/glove/vectors.txt', binary=False)  # Update path
        embeddings = []
        for sentence in tokenized_sentences:
            sentence_embeddings = [glove_model[word] for word in sentence if word in glove_model]
            if sentence_embeddings:
                embeddings.append(np.mean(sentence_embeddings, axis=0))
        return np.vstack(embeddings)

    else:
        raise ValueError("Invalid embedding method. Choose 'word2vec', 'bert', 'gpt', or 'glove'.")

def main():
    """
    Main function to extract text from a PDF, apply OCR, tokenize the text,
    and save the results.
    """
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_pdf_file> <tokenization_method> <embedding_method>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    tokenization_method = sys.argv[2]
    embedding_method = sys.argv[3]
    output_path = pdf_path.rsplit('.', 1)[0] + "_embedding"

    extracted_text, extracted_images = extract_text_and_images_from_pdf(pdf_path)
    print("Extracted Text:", extracted_text[:500])  
    print("Number of Images Extracted:", len(extracted_images))  

    ocr_results = apply_ocr_to_images(extracted_images)
    print("OCR Results:", ocr_results[:3])  
    tokenized_sentences = tokenize_text(extracted_text, tokenization_method)  
    print("Tokenized Sentences:", tokenized_sentences[:3])  
    embeddings = generate_embeddings(tokenized_sentences, embedding_method)  

    if embeddings is not None:
        np.save(output_path, embeddings)  
        print("Embeddings saved to", output_path)
    else:
        print("No embeddings generated.")

if __name__ == "__main__":
    main()
