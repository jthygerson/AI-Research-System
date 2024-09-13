# decision_maker.py

import openai
import os
import logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def decide_next_step(results, experiment_plan):
    prompt = (
        f"The experiment yielded the following results: {results}.\n"
        f"The target test accuracy is 95%.\n"
        f"Based on these outcomes and the target, should we:\n"
        f"1. Refine the experiment further\n"
        f"2. Redesign the experiment\n"
        f"3. Generate new research ideas\n"
        f"4. Proceed to augment the AI Research System\n"
        f"Provide your recommendation as 'refine', 'redesign', 'new_idea', or 'proceed', "
        f"and briefly explain your reasoning."
    )

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )

        decision_text = response.choices[0].text.strip().lower()
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
