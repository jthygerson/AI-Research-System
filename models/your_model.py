# your_model.py

import logging
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import random

def run_model(parameters):
    try:
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Set random seeds for reproducibility
        seed = int(parameters.get('seed', 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        # Load dataset
        dataset_name = parameters.get('selected_dataset', 'ag_news')
        raw_datasets = load_dataset(dataset_name)
        logging.info(f"Loaded dataset: {dataset_name}")

        # Preprocess data
        tokenizer_name = parameters.get('model_architecture', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logging.info(f"Using tokenizer: {tokenizer_name}")

        def tokenize_function(example):
            return tokenizer(example['text'], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        # Prepare training and evaluation datasets
        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets['test']

        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # Determine number of labels
        num_labels = len(set(train_dataset['label']))
        logging.info(f"Number of labels: {num_labels}")

        # Load pre-trained model
        model_name = parameters.get('model_architecture', 'distilbert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)
        logging.info(f"Loaded model: {model_name}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=int(parameters.get('num_epochs', 3)),
            per_device_train_batch_size=int(parameters.get('batch_size', 8)),
            per_device_eval_batch_size=int(parameters.get('batch_size', 8)),
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            seed=seed,
        )

        # Define compute metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            accuracy = (predictions == labels).mean()
            return {'accuracy': accuracy}

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Start training
        logging.info("Starting training...")
        trainer.train()

        # Evaluate the model
        logging.info("Evaluating the model...")
        eval_results = trainer.evaluate()

        results = {
            'test_accuracy': eval_results['eval_accuracy'],
            'model_parameters': parameters
        }
        logging.info(f"Experiment Results: {results}")
        return results

    except Exception as e:
        logging.error(f"Error in run_model: {e}")
        return None
