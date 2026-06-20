# Text Summarization

Abstractive text summarization built on the **mT5** (`google/mt5-small`)
sequence-to-sequence model from Hugging Face `transformers`, fine-tuned on the
[`ccdv/arxiv-summarization`](https://huggingface.co/datasets/ccdv/arxiv-summarization)
dataset to summarize scientific articles.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All settings live in `config.ini`:

```ini
[dataset]
dataset_name=ccdv/arxiv-summarization
local_data_path=data/raw/

[model]
model_checkpoint=google/mt5-small
save_dir=models/

[preprocess]
max_input_length=1024
max_target_length=128

[train]
epoch=100
batch_size=8
learning_rate=2e-5
```

## Usage

### Download the dataset
Downloads the dataset and caches it locally under `local_data_path`:

```bash
python data/data_loader.py
```

### Train
```bash
python -m src.train
```

The fine-tuned model and tokenizer are saved to `save_dir` (`models/`) after
each epoch, and ROUGE scores are reported on the validation split.

## Project structure

```
text-summarization/
├── config.ini                  # dataset, model, preprocessing and training settings
├── data/
│   └── data_loader.py          # download / load the dataset
├── src/
│   ├── model_and_tokenizer.py  # load and cache the mT5 model and tokenizer
│   ├── utils.py                # config, preprocessing and ROUGE metrics
│   ├── summarizer.py           # inference wrapper (work in progress)
│   └── train.py                # training loop
├── notebooks/                  # exploration notebooks
├── app.py                      # inference entry point (work in progress)
├── requirements.txt
├── LICENSE
└── README.md
```

## License

Released under the [MIT License](LICENSE).
