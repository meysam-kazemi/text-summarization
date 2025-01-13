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