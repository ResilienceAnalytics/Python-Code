# Importing necessary libraries
import pdfplumber  # Used for extracting text from PDF files
import nltk  # Natural Language Toolkit for text processing
from nltk.tokenize import word_tokenize  # For tokenizing text into words
from transformers import BertTokenizer  # Advanced tokenizer from Hugging Face
from PIL import Image  # Python Imaging Library for image processing
import io  # Input/output module for handling byte streams
import re  # Regular expressions library
import sys  # System-specific parameters and functions

# Download and initialize the NLTK tokenizer
nltk.download('punkt')

# Initialize the BERT tokenizer
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

# Define a function to extract text and images from a PDF
def extract_text_and_images_from_pdf(pdf_path):
    """
    Extract text and images from the given PDF file.

    Args:
        pdf_path (str): File path of the PDF.

    Returns:
        tuple: A tuple containing the extracted text and a list of images.
    """
    try:
        text = ""  # Initialize an empty string for text
        images = []  # Initialize an empty list for images
        with pdfplumber.open(pdf_path) as pdf:  # Open the PDF
            for page in pdf.pages:  # Iterate through each page
                page_text = page.extract_text()  # Extract text from the current page
                if page_text:  # If text is found
                    text += page_text + "\n"  # Append it to the text string
                images.extend(page.images)  # Append images from the current page
        return text, images  # Return the extracted text and images
    except Exception as e:
        raise RuntimeError(f"Error while extracting PDF: {e}")

# Define a function to apply OCR to images
def apply_ocr_to_images(images):
    """
    Apply Optical Character Recognition (OCR) on the list of images.

    Args:
        images (list): List of images to process.

    Returns:
        list: A list of text strings extracted from the images.
    """
    try:
        ocr_results = []  # Initialize an empty list for OCR results
        for img in images:  # Iterate through each image
            im = Image.open(io.BytesIO(img['stream']))  # Open the image
            ocr_results.append(pytesseract.image_to_string(im))  # Apply OCR and append the result
        return ocr_results  # Return the OCR results
    except Exception as e:
        raise RuntimeError(f"Error while applying OCR: {e}")

# Define a function to tokenize text based on the chosen method
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

# Define the main function for script execution
def main():
    """
    Main function to extract text from a PDF, apply OCR, tokenize the text,
    and save the results.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_pdf_file> <tokenization_method>")
        sys.exit(1)

    pdf_path = sys.argv[1]  # Get the PDF file path from command line arguments
    method = sys.argv[2]  # Get the tokenization method
    output_path = pdf_path.rsplit('.', 1)[0] + "_tokenizer.txt"  # Output file path

    # Extract text and images from the PDF
    extracted_text, extracted_images = extract_text_and_images_from_pdf(pdf_path)

    # Apply OCR to the images
    ocr_results = apply_ocr_to_images(extracted_images)

    # Tokenize the extracted text
    tokenized_text = tokenize_text(extracted_text, method)

    # Save the tokenized content to the output file
    with open(output_path, 'w') as f:
        for line in tokenized_text:
            f.write(' '.join(line) + '\n')
        for line in ocr_results:
            f.write(normalize_text(line) + '\n')

    print(f"Tokenized content written to {output_path}")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
