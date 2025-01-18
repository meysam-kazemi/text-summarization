# src/utils.py
import os
import numpy as np
from datasets import load_dataset
import configparser
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

def preprocess(
    examples,
    tokenizer,
    max_input_length,
    max_target_length
    ):
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["abstract"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


