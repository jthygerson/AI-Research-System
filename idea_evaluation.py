# idea_evaluation.py

import openai
import os
import logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def evaluate_ideas(ideas):
    best_idea = None
    highest_score = 0

    for idea in ideas:
        prompt = (
            f"Evaluate the following idea on a scale of 1-10 for novelty and probability "
            f"of success, considering current AI research trends.\nIdea: {idea}\n"
            "Provide your evaluation in the format: Novelty: X, Success Probability: Y."
        )

        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.5,
            )

            scores_text = response.choices[0].text.strip()
            novelty_score, success_score = parse_scores(scores_text)

            total_score = novelty_score + success_score
            if total_score > highest_score:
                highest_score = total_score
                best_idea = idea

            logging.info(f"Idea: {idea}, Novelty: {novelty_score}, Success: {success_score}")

        except Exception as e:
            logging.error(f"Error evaluating idea '{idea}': {e}")
            continue

    return best_idea

def parse_scores(scores_text):
    try:
        parts = scores_text.replace(" ", "").split(',')
        novelty = int(parts[0].split(':')[1])
        success = int(parts[1].split(':')[1])
        return novelty, success
    except Exception as e:
        logging.error(f"Error parsing scores '{scores_text}': {e}")
        return 0, 0
