import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from PIL import Image
import io
import re
import sys

# Initialisation des tokenizers
nltk.download('punkt')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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


class DomainClassifier:
    """
    A classifier that categorizes terms into 'Natural Sciences' or 'Human Sciences'.

    Attributes:
        dataset (dict): A dictionary mapping terms to their corresponding domains.
    """
    def __init__(self):
        self.dataset = {
            # Terms for Creation-Natural Science sequence
            "creation": "Natural Sciences", "manufacture": "Natural Sciences",
            "elaboration": "Natural Sciences", "invention": "Natural Sciences",
            "conception": "Natural Sciences", "genesis": "Natural Sciences",
            "initiation": "Natural Sciences", "production": "Natural Sciences",
            "fabrication": "Natural Sciences", "assemblage": "Natural Sciences",
            "construction": "Natural Sciences", "development": "Natural Sciences",
            "refinement": "Natural Sciences", "expansion": "Natural Sciences",
            "amplification": "Natural Sciences",

            # Terms for Distribution-Human Science sequence
            "distribution": "Human Sciences", "allocation": "Human Sciences",
            "ventilation": "Human Sciences", "dissemination": "Human Sciences",
            "dispersion": "Human Sciences", "apportionment": "Human Sciences",
            "diffusion": "Human Sciences", "assignation": "Human Sciences",
            "allotment": "Human Sciences", "rationing": "Human Sciences",
            "designation": "Human Sciences", "circulation": "Human Sciences",
            "broadcasting": "Human Sciences", "dispersal": "Human Sciences",
            "propagation": "Human Sciences",

            # Terms for Use-Natural Science sequence
            "use": "Natural Sciences", "functioning": "Natural Sciences",
            "usage": "Natural Sciences"
        }

    def classify_term(self, term):
        """
        Classify a given term according to its domain.

        Args:
            term (str): The term to classify.

        Returns:
            str: The domain of the term.
        """
        return self.dataset.get(term, "Unknown")

def tokenize_text(text, method):
    """
    Tokenize the text using tokenizer BERT.
    """
    text = normalize_text(text)
    lines = text.split('\n')
    tokenized_content = [tokenizer.tokenize(line) for line in lines]
    return tokenized_content

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

def tokenize_text(text, method):
    """
    Tokenize the given text based on the specified method.

    Args:
        text (str): Text to tokenize.
        method (str): Method of tokenization ('word', 'subword', 'letter').

    Returns:
        list: A list of tokenized content.
    """
    try:
        text = normalize_text(text)  # Normalize the text first
        lines = text.split('\n')  # Split the text into lines

        # Tokenize each line based on the chosen method
        if method == 'word':
            tokenized_content = [word_tokenize(line) for line in lines]
        elif method == 'subword':
            tokenized_content = [bert_tokenizer.tokenize(line) for line in lines]
        elif method == 'letter':
            tokenizer = RegexpTokenizer(r'\s+', gaps=True)
            tokenized_content = [list(tokenizer.tokenize(line)) for line in lines]
        else:
            raise ValueError("Invalid tokenization method. Choose 'word', 'subword', or 'letter'.")

        return tokenized_content  # Return the tokenized content
    except Exception as e:
        raise RuntimeError(f"Error while tokenizing: {e}")

def tokenize_and_classify(text, method, classifier):
    """
    Tokenize and classify terms in the given text.

    Args:
        text (str): The text to process.
        classifier (DomainClassifier): The classifier to use for categorizing terms.

    Returns:
        list: A list of tuples where each tuple contains a token and its domain.
    """
    tokenized_content = tokenize_text(text, method)
    classified_tokens = [(token, classifier.classify_term(token)) for line in tokenized_content for token in line]
    return classified_tokens

def main():
    """
    Main function to extract text from a PDF, apply OCR, tokenize the text,
    and save the results.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_pdf_file> <tokenization_method>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    method = sys.argv[2]
    output_path = pdf_path.rsplit('.', 1)[0] + "_tokenized_classified.txt"

    extracted_text, extracted_images = extract_text_and_images_from_pdf(pdf_path)
    ocr_results = apply_ocr_to_images(extracted_images)
    classifier = DomainClassifier()
    tokenized_classified_text = tokenize_and_classify(extracted_text + ' '.join(ocr_results), method, classifier)

    with open(output_path, 'w') as f:
        for token, domain in tokenized_classified_text:
            f.write(f"{token} - {domain}\n")

    print(f"Tokenized and classified content written to {output_path}")

if __name__ == "__main__":
    main()
