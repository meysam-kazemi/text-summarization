import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler, DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_and_tokenizer import modelTokenizer
from data.data_loader import data_loader
from src.utils import (
    read_config,
    Preprocess,
    Metrics,
    postprocess_text
)


config = read_config()
data = data_loader(config)
mt = modelTokenizer(config)
metrics = Metrics(mt.tokenizer)
preprocess = Preprocess(
    mt.tokenizer,
    int(config['preprocess']['max_input_length']),
    int(config['preprocess']['max_target_length']),
)

num_train_epochs = int(config['train']['epoch'])
batch_size = int(config['train']['batch_size'])
learning_rate = float(config['train']['learning_rate'])

tokenized_datasets = preprocess(data)


data_collator = DataCollatorForSeq2Seq(mt.tokenizer, model=mt.model)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=batch_size
)


optimizer = AdamW(mt.model.parameters(), lr=learning_rate)


accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    mt.model, optimizer, train_dataloader, eval_dataloader
)


num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#########################################################3


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