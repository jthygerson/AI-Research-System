# decision_maker.py

import openai
import os
import logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def decide_next_step(results, experiment_plan):
    target_accuracy = 0.90  # Adjust as per your goal

    prompt = (
        f"The experiment yielded the following results: {results}.\n"
        f"The target test accuracy is {target_accuracy * 100}%.\n"
        "Based on these outcomes and the target, should we:\n"
        "1. Refine the experiment further\n"
        "2. Redesign the experiment\n"
        "3. Generate new research ideas\n"
        "4. Proceed to augment the AI Research System\n"
        "Provide your recommendation as 'refine', 'redesign', 'new_idea', or 'proceed', "
        "and briefly explain your reasoning."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps make decisions based on experiment results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.5,
        )

        decision_text = response['choices'][0]['message']['content'].strip().lower()
        logging.info(f"Decision text: {decision_text}")

        if 'refine' in decision_text:
            return 'refine'
        elif 'redesign' in decision_text:
            return 'redesign'
        elif 'new idea' in decision_text or 'new_idea' in decision_text:
            return 'new_idea'
        elif 'proceed' in decision_text:
            return 'proceed'
        else:
            return None

    except Exception as e:
        logging.error(f"Error in decision making: {e}")
        return None
