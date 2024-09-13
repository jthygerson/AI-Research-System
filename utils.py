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
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def parse_experiment_plan(experiment_plan):
    parameters = {}

    # Extract sections using regular expressions
    sections = re.split(r'^\d+\.\s+', experiment_plan, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    # Map section titles to keys
    section_titles = ['Objective', 'Methodology', 'Suggested Datasets', 'Model Architecture', 'Hyperparameters', 'Evaluation Metrics']

    for section in sections:
        for title in section_titles:
            if section.startswith(title):
                content = section[len(title):].strip(':').strip()
                parameters[title.lower().replace(' ', '_')] = content
                break

    # Process hyperparameters into a dictionary
    if 'hyperparameters' in parameters:
        parameters['hyperparameters'] = parse_hyperparameters(parameters['hyperparameters'])

    # Process datasets into a list
    if 'suggested_datasets' in parameters:
        parameters['datasets'] = parse_datasets(parameters['suggested_datasets'])

    # Extract model architecture
    if 'model_architecture' in parameters:
        parameters['model_architecture'] = parameters['model_architecture'].split('\n')[0].strip()

    return parameters

def parse_hyperparameters(hyperparams_text):
    hyperparams = {}
    for param in hyperparams_text.split('\n'):
        if ':' in param:
            key, value = param.split(':', 1)
            hyperparams[key.strip()] = value.strip()
    return hyperparams

def parse_datasets(datasets_text):
    datasets = [dataset.strip('-• ').strip() for dataset in datasets_text.split('\n') if dataset.strip()]
    return datasets
