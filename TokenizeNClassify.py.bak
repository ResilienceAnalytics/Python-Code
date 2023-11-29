import nltk
from nltk.tokenize import word_tokenize
import sys

class DomainClassifier:
    """
    A classifier that categorizes terms into 'Natural Sciences' or 'Human Sciences'.

    Attributes:
        dataset (dict): A dictionary mapping terms to their corresponding domain.
    """

    def __init__(self):
        self.dataset = {
            # Creation-Natural Science terms
            "creation": "Natural Sciences", "manufacture": "Natural Sciences",
            "elaboration": "Natural Sciences", "invention": "Natural Sciences",
            "conception": "Natural Sciences", "genesis": "Natural Sciences",
            "initiation": "Natural Sciences", "production": "Natural Sciences",
            "fabrication": "Natural Sciences", "assemblage": "Natural Sciences",
            "construction": "Natural Sciences", "development": "Natural Sciences",
            "refinement": "Natural Sciences", "expansion": "Natural Sciences",
            "amplification": "Natural Sciences",

            # Distribution-Human Science terms
            "distribution": "Human Sciences", "allocation": "Human Sciences",
            "ventilation": "Human Sciences", "dissemination": "Human Sciences",
            "dispersion": "Human Sciences", "apportionment": "Human Sciences",
            "diffusion": "Human Sciences", "assignation": "Human Sciences",
            "allotment": "Human Sciences", "rationing": "Human Sciences",
            "designation": "Human Sciences", "circulation": "Human Sciences",
            "broadcasting": "Human Sciences", "dispersal": "Human Sciences",
            "propagation": "Human Sciences",

            # Use-Natural Science terms
            "use": "Natural Sciences", "functioning": "Natural Sciences",
            "usage": "Natural Sciences"
        }

    def classify_term(self, term):
        """
        Classify a term based on the domain.

        Args:
            term (str): The term to classify.

        Returns:
            str: The domain of the term.
        """
        return self.dataset.get(term, "Unknown")

def tokenize_and_classify(text, classifier):
    """
    Tokenize the text and classify each token based on the domain.

    Args:
        text (str): The text to tokenize and classify.
        classifier (DomainClassifier): The classifier to use for categorizing terms.

    Returns:
        list: A list of tuples where each tuple contains a token and its domain.
    """
    tokens = word_tokenize(text)
    classified_tokens = [(token, classifier.classify_term(token)) for token in tokens]
    return classified_tokens

def main():
    """
    Main function to read text from command line, tokenize it, and classify each token.
    """
    if len(sys.argv) != 2:
        print("Usage: python TokenizeNClassify.py <input_text>")
        sys.exit(1)

    text = sys.argv[1]
    classifier = DomainClassifier()
    classified_tokens = tokenize_and_classify(text, classifier)

    for token, domain in classified_tokens:
        if domain != "Unknown":
            print(f"{token}: {domain}")

if __name__ == "__main__":
    nltk.download('punkt')
    main()
