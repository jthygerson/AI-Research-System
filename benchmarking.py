# benchmarking.py

import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
import torch

def run_benchmarks():
    try:
        # Load a standard benchmarking dataset
        dataset = load_dataset('glue', 'sst2')
        logging.info("Loaded GLUE SST-2 dataset for benchmarking.")

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

        # Preprocess the data
        def tokenize_function(example):
            return tokenizer(
                example['sentence'],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(
            [col for col in tokenized_datasets['train'].column_names if col not in ['input_ids', 'attention_mask', 'label']]
        )
        tokenized_datasets.set_format('torch')

        # Prepare training and evaluation datasets
        train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))  # Use a subset for quick benchmarking
        eval_dataset = tokenized_datasets['validation'].shuffle(seed=42).select(range(200))

        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logging.info(f"Using device: {device}")

        training_args = TrainingArguments(
            output_dir='./benchmark_results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            logging_dir='./benchmark_logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        # Define compute_metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            accuracy = (predictions == labels).mean()
            return {'accuracy': accuracy}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        logging.info(f"Benchmarking results: {eval_results}")

        return eval_results

    except Exception as e:
        logging.error(f"Error in benchmarking: {e}")
        return None
