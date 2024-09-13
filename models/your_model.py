# models/your_model.py

import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

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
            return tokenizer(example['text'], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Prepare training and evaluation datasets
        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets.get('validation') or tokenized_datasets.get('test')

        # Set hyperparameters from parameters or defaults
        hyperparams = parameters.get('hyperparameters', {})
        num_train_epochs = float(hyperparams.get('epochs', 3))
        per_device_train_batch_size = int(hyperparams.get('batch_size', 8))

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
