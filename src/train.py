import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_and_tokenizer import modelTokenizer
from data.data_loader import data_loader
from src.utils import (
    read_config,
    preprocess,
    postprocess,
)


config = read_config()
data = data_loader(config)
mt = modelTokenizer(config)
preprocess = preprocess(
    mt.tokenizer,
    config['preprocess']['max_input_length'],
    config['preprocess']['max_target_length'],
)

tokenized_datasets = preprocess(data)

#########################################################3

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,tokenize_datasets

optimizer = AdamW(mt.model.parameters(), lr=2e-5)


accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    mt.model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# --------------
## Training loop
# --------------

progress_bar = tqdm(range(num_training_steps))
output_dir = config['model']['save_dir']
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = mt.postprocess(predictions_gathered, labels_gathered)
        mt.eval.add_batch(predictions=true_predictions, references=true_labels)

    results = mt.eval.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        mt.tokenizer.save_pretrained(output_dir)