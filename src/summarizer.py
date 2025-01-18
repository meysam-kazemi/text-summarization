import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import read_config
from src.model_and_tokenizer import modelAndTokenizer

class Summarizer:
    def __init__(self, config):
        self.mt = modelAndTokenizer(config)

    