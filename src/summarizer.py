import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_config, preprocess
from src.model_and_tokenizer import modelAndTokenizer

class Summarizer:
    def __init__(self, config):
        self.mt = modelAndTokenizer(config)
        self.max_input_length = config['preprocess']['max_input_length']
        self.max_target_length = config['preprocess']['max_target_length']

    def __call__(self, dataset):
        pass
    