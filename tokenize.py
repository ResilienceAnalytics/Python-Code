import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
import pytesseract
from PIL import Image
import io
import re
import sys

# Download NLTK tokenizer
nltk.download('punkt')

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

def tokenize_text(text):
    """
    Tokenizes the text into words.

    Args:
    text (str): The text to be tokenized.

    Returns:
    list: A list of lists, where each sublist contains tokens from one line of the text.
    """
    lines = text.split('\n')
    tokenized_content = [word_tokenize(line) for line in lines]
    return tokenized_content

def main():
    """
    Main function to extract, OCR, tokenize text from a PDF file, and save the output.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = pdf_path.rsplit('.', 1)[0] + "_tokenizer.txt"

    extracted_text, extracted_images = extract_text_and_images_from_pdf(pdf_path)
    ocr_results = apply_ocr_to_images(extracted_images)
    tokenized_text = tokenize_text(extracted_text)

    with open(output_path, 'w') as f:
        for line in tokenized_text:
            f.write(' '.join(line) + '\n')
        for line in ocr_results:
            f.write(line + '\n')

    print(f"Tokenized content written to {output_path}")

if __name__ == "__main__":
    main()
