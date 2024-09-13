# models/your_model.py

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

def run_model(experiment_plan, parameters):
    try:
        # Use the selected dataset
        selected_dataset = parameters.get('selected_dataset')
        if not selected_dataset:
            logging.error("No dataset selected for the experiment.")
            return None

        # Load the dataset
        dataset = load_dataset(selected_dataset)
        logging.info(f"Loaded dataset: {selected_dataset}")

        # Use the model architecture from parameters or default
        model_name = parameters.get('model_architecture', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Preprocess the data
        def tokenize_function(example):
            return tokenizer(
                example['text'],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Remove columns not needed
        tokenized_datasets = tokenized_datasets.remove_columns(
            [col for col in tokenized_datasets['train'].column_names if col not in ['input_ids', 'attention_mask', 'label']]
        )

        # Set format for PyTorch
        tokenized_datasets.set_format('torch')

        # Prepare training and evaluation datasets
        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets.get('validation') or tokenized_datasets.get('test')

        if eval_dataset is None:
            logging.error("No validation or test dataset found.")
            return None

        # Set hyperparameters from parameters or defaults
        hyperparams = parameters.get('hyperparameters', {})
        num_train_epochs = float(hyperparams.get('epochs', 3))
        per_device_train_batch_size = int(hyperparams.get('batch_size', 8))

        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logging.info(f"Using device: {device}")

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch",
            logging_dir='./logs',
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

        # Initialize Trainer
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
        test_accuracy = eval_results.get('eval_accuracy', None)
        logging.info(f"Evaluation results: {eval_results}")

        results = {
            'test_accuracy': test_accuracy,
            'eval_results': eval_results
        }
        return results

    except Exception as e:
        logging.error(f"Error in run_model: {e}")
        return None
