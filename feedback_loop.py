# feedback_loop.py

import openai
import os
import logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def refine_experiment(experiment_plan, results):
    prompt = (
        f"The experiment was conducted as per the following plan:\n{experiment_plan}\n"
        f"The results obtained are: {results}\n"
        "Based on these results, suggest improvements to enhance performance. "
        "Update the experiment plan accordingly. Ensure the updated plan has the same sections as before."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that refines experiments based on results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            n=1,
            temperature=0.7,
        )

        refined_plan = response['choices'][0]['message']['content'].strip()
        logging.info(f"Refined Experiment Plan:\n{refined_plan}")
        return refined_plan

    except Exception as e:
        logging.error(f"Error refining experiment: {e}")
        return experiment_plan  # Return the original plan if refinement fails
