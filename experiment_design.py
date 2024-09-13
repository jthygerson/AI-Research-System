# experiment_design.py

import openai
import os
import logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def design_experiment(idea):
    prompt = (
        f"Design a detailed experiment to test the following idea:\nIdea: {idea}\n"
        "Include the objective, methodology, datasets required (preferably open-source), "
        "model architectures to use, hyperparameters to consider, and evaluation metrics."
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
        return experiment_plan

    except Exception as e:
        logging.error(f"Error designing experiment: {e}")
        return None
