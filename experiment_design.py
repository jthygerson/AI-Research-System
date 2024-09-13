# experiment_design.py

import openai
import os
import logging
from utils import parse_experiment_plan
from dataset_search import search_datasets

openai.api_key = os.getenv('OPENAI_API_KEY')

def design_experiment(idea):
    prompt = (
        f"Design a detailed experiment to test the following idea:\nIdea: {idea}\n"
        "Provide the experiment plan with the following sections:\n"
        "1. Objective\n"
        "2. Methodology\n"
        "3. Suggested Datasets (specify dataset names or sources)\n"
        "4. Model Architecture (specify model types)\n"
        "5. Hyperparameters (list them as key-value pairs)\n"
        "6. Evaluation Metrics\n"
        "Ensure that each section is clearly labeled."
    )

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        experiment_plan = response.choices[0].text.strip()
        logging.info(f"Experiment Plan:\n{experiment_plan}")

        # Parse the experiment plan
        parameters = parse_experiment_plan(experiment_plan)
        suggested_datasets = parameters.get('datasets', [])

        # Search for datasets on Hugging Face
        available_datasets = []
        for dataset_name in suggested_datasets:
            datasets = search_datasets(dataset_name)
            if datasets:
                available_datasets.extend(datasets)

        if available_datasets:
            selected_dataset = available_datasets[0]
            logging.info(f"Selected dataset: {selected_dataset}")
            parameters['selected_dataset'] = selected_dataset
        else:
            logging.warning("No relevant datasets found on Hugging Face.")
            parameters['selected_dataset'] = None

        return experiment_plan, parameters

    except Exception as e:
        logging.error(f"Error designing experiment: {e}")
        return None, {}
