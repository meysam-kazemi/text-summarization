import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler, DataCollatorForSeq2Seq
from tqdm.auto import tqdm
import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_and_tokenizer import modelAndTokenizer
from data.data_loader import dataLoader
from src.utils import (
    read_config,
    Preprocess,dataLoader
    Metrics,
    postprocess_text
)


config = read_config()
data = dataLoader(config)
mt = modelAndTokenizer(config)
metrics = Metrics(mt.tokenizer)
preprocess = Preprocess(
    mt.tokenizer,
    int(config['preprocess']['max_input_length']),
    int(config['preprocess']['max_target_length']),
)

num_train_epochs = int(config['train']['epoch'])
batch_size = int(config['train']['batch_size'])
learning_rate = float(config['train']['learning_rate'])
output_dir = config['model']['save_dir']

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

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=mt.tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # If we did not pad to max length, we need to pad the labels too
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=mt.tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, mt.tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = mt.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = mt.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            metrics.rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Compute metrics
    result = metrics.rouge_score.compute()
    # Extract the median ROUGE scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        mt.tokenizer.save_pretrained(output_dir)