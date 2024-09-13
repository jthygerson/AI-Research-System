# benchmarking.py

import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_benchmarks():
    try:
        # Load a standard benchmarking dataset
        dataset = load_dataset('glue', 'mrpc')
        logging.info("Loaded GLUE MRPC dataset for benchmarking.")

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

        def tokenize_function(example):
            return tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets['validation']

        training_args = TrainingArguments(
            output_dir='./benchmark_results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch",
            logging_dir='./benchmark_logs',
            logging_steps=10,
        )

        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            labels = p.label_ids
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

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
