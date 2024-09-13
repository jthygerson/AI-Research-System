# experiment_execution.py

import logging
from models.your_model import run_model

def execute_experiment(experiment_plan, parameters):
    # Pass the parameters to run_model
    try:
        results = run_model(experiment_plan, parameters)
        return results
    except Exception as e:
        logging.error(f"Error executing experiment: {e}")
        return None
