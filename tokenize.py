import pdfplumber
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pytesseract
from transformers import BertTokenizer
from PIL import Image
import io
import re
import sys

# Download NLTK tokenizer
nltk.download('punkt')

# Initialize BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def normalize_text(text):
    """
    Normalize text by converting it to lowercase and removing non-standard punctuation.
    
    Args:
        text (str): The text to normalize.

    Returns:
        str: Normalized text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_text_and_images_from_pdf(pdf_path):
    """
    Extract text and images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        tuple: Returns a tuple containing the extracted text and a list of images.
    """
    try:
        text = ""
        images = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                images.extend(page.images)
        return text, images
    except Exception as e:
        raise RuntimeError(f"Error while extracting PDF: {e}")

def apply_ocr_to_images(images):
    """
    Apply Optical Character Recognition (OCR) to a list of images.

    Args:
        images (list): List of images.

    Returns:
        list: List of strings resulting from OCR.
    """
    try:
        ocr_results = []
        for img in images:
            im = Image.open(io.BytesIO(img['stream']))
            ocr_results.append(pytesseract.image_to_string(im))
        return ocr_results
    except Exception as e:
        raise RuntimeError(f"Error while applying OCR: {e}")

def tokenize_text(text, method):
    """
    Tokenize text based on the chosen method.

    Args:
        text (str): The text to tokenize.
        method (str): Tokenization method ('word', 'subword', 'letter').

    Returns:
        list: List of tokens.
    """
    try:
        text = normalize_text(text)
        lines = text.split('\n')

        if method == 'word':
            tokenized_content = [word_tokenize(line) for line in lines]
        elif method == 'subword':
            tokenized_content = [bert_tokenizer.tokenize(line) for line in lines]
        elif method == 'letter':
            tokenizer = RegexpTokenizer(r'\s+', gaps=True)
            tokenized_content = [list(tokenizer.tokenize(line)) for line in lines]
        else:
            raise ValueError("Invalid tokenization method. Choose 'word', 'subword', or 'letter'.")

        return tokenized_content
    except Exception as e:
        raise RuntimeError(f"Error while tokenizing: {e}")

def main():
    """
    Main function to extract, apply OCR, tokenize text from a PDF file, and save the result.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_pdf_file> <tokenization_method>")
        sys.exit(1)

    try:
        pdf_path = sys.argv[1]
        method = sys.argv[2]
        output_path = pdf_path.rsplit('.', 1)[0] + "_tokenizer.txt"

        extracted_text, extracted_images = extract_text_and_images_from_pdf(pdf_path)
        ocr_results = apply_ocr_to_images(extracted_images)
        tokenized_text = tokenize_text(extracted_text, method)

        with open(output_path, 'w') as f:
            for line in tokenized_text:
                f.write(' '.join(line) + '\n')
            for line in ocr_results:
                f.write(normalize_text(line) + '\n')

        print(f"Tokenized content written to {output_path}")
    except Exception as e:
        print(f"Error during script execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
