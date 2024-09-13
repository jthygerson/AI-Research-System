# utils.py

import logging
import re
import os

def initialize_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        filename='logs/system.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

def parse_experiment_plan(experiment_plan):
    parameters = {}

    # Extract model architecture
    match = re.search(r'Model Architecture:\s*(.+?)(?:\n\n|$)', experiment_plan, re.DOTALL)
    if match:
        parameters['model_architecture'] = match.group(1).strip()

    # Extract hyperparameters
    match = re.search(r'Hyperparameters:\s*(.+?)(?:\n\n|$)', experiment_plan, re.DOTALL)
    if match:
        hyperparams_text = match.group(1).strip()
        parameters['hyperparameters'] = parse_hyperparameters(hyperparams_text)

    # Extract suggested datasets
    match = re.search(r'Suggested Datasets:\s*(.+?)(?:\n\n|$)', experiment_plan, re.DOTALL)
    if match:
        datasets_text = match.group(1).strip()
        parameters['datasets'] = parse_datasets(datasets_text)

    # Extract evaluation metrics
    match = re.search(r'Evaluation Metrics:\s*(.+?)(?:\n\n|$)', experiment_plan, re.DOTALL)
    if match:
        parameters['evaluation_metrics'] = [metric.strip() for metric in match.group(1).split(',')]

    return parameters

def parse_hyperparameters(hyperparams_text):
    hyperparams = {}
    for param in hyperparams_text.split('\n'):
        if ':' in param:
            key, value = param.split(':', 1)
            hyperparams[key.strip()] = value.strip()
    return hyperparams

def parse_datasets(datasets_text):
    datasets = [dataset.strip('- ').strip() for dataset in datasets_text.split('\n') if dataset.strip()]
    return datasets
