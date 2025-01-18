import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_config

config = read_config()
model_checkpoint = config['model']['model_checkpoint']
save_dir = config['model']['save_dir']


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)




class modelAndTokenizer:
    def __init__(self, config):
        self.save_dir = config['model']['save_dir']
        self.model_name = config['model']['model_checkpoint']


        try:
            self._load_model_and_tokenizer_locally()
            print("Locally")
        except:
            self._load_and_save_model()

    def _load_and_save_model(self):
        """
        Loads a pre-trained model and tokenizer, and saves them to the specified directory.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Load the model and tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Save the model and tokenizer
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)

        print(f"Model and tokenizer for '{self.model_name}' saved in '{self.save_dir}'.")

    def _load_model_and_tokenizer_locally(self):
        """
        Loads the model and tokenizer from the specified directory.
        """

        # Load the model and tokenizer from the specified directory
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.save_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
    