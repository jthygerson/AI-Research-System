# experiment_execution.py

import logging
from models.your_model import run_model

def execute_experiment(experiment_plan):
    # Parse the experiment plan to extract actionable steps
    # For simplicity, we'll assume 'run_model' handles execution based on the plan
    try:
        results = run_model(experiment_plan)
        return results
    except Exception as e:
        logging.error(f"Error executing experiment: {e}")
        return None
