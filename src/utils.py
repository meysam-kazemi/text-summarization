# src/utils.py
import os
import numpy as np
from datasets import load_dataset
import configparser
import evaluate
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForTokenClassification


def read_config(config_file='config.ini'):
    """
    Reads the configuration file and returns the settings as a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    config = configparser.ConfigParser()
    
    # Check if the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file '{config_file}' does not exist.")
    
    config.read(config_file)
    
    # Convert config sections to a dictionary
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    
    return config_dict

class Preprocess:
    def __init__(
        self,
        tokenizer,
        max_input_length,
        max_target_length
        ):

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
 
    def _preprocess_func(self, examples):
        model_inputs = self.tokenizer(
            examples["article"],
            max_length=self.max_input_length
            truncation=True,
        )
        labels = self.tokenizer(
            examples["abstract"], max_length=self.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def _tokenize_data(self, dataset):
        return dataset.map(self._preprocess_func, batched=True)
    
    def __call__(self, dataset):
        tokenized_datasets = self._tokenize_data(dataset)
        tokenized_datasets.set_format("torch")
        return tokenized_datasets



class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge_score = evaluate.load("rouge")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = self.rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}