import json
import sys
import os
from transformers import BertTokenizer, BertForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader

class JsonDataset(Dataset):
    """
    A Dataset class to handle the JSON data for NLP model training.
    """
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = []

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for tokens_list in data:
                    # Convert the list of tokens into a string
                    sentence = ' '.join(tokens_list)
                    self.sentences.append(sentence)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            sys.exit(1)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer(sentence, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}

def train_model(model, dataset, epochs=1, save_path='model_save'):
    """
    Function to train the NLP model.
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch['input_ids']
            masks = batch['attention_mask']

            outputs = model(inputs, attention_mask=masks, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        # Save the model after each epoch
        model.save_pretrained(os.path.join(save_path, f"epoch_{epoch}"))
        print(f"Model saved for epoch {epoch} at {save_path}")

def main():
    """
    Main function to handle model training.
    """
    if len(sys.argv) != 3:
        print("Usage: python Training.py <path_to_json_file> <model_type>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_choice = sys.argv[2].lower()

    if model_choice == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    elif model_choice == 'gpt-2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        print("Invalid model choice. Please choose 'bert' or 'gpt-2'.")
        sys.exit(1)

    dataset = JsonDataset(file_path, tokenizer, max_length=512)
    train_model(model, dataset, epochs=3)

if __name__ == "__main__":
    main()
