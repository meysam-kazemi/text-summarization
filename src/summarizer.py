import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_config, preprocess
from src.model_and_tokenizer import modelAndTokenizer

class Summarizer:
    def __init__(self, config):
        self.mt = modelAndTokenizer(config)

    def __call__(self, dataset):
        pass
    