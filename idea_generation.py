# idea_generation.py

import openai
import os
from utils import initialize_logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_ideas():
    prompt = (
        "Generate five novel research ideas in the field of AI that can be executed "
        "with limited computational resources (e.g., single GPU) and within one week. "
        "List them as bullet points."
    )

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=250,
            n=1,
            stop=None,
            temperature=0.7,
        )

        ideas_text = response.choices[0].text.strip()
        ideas = [idea.strip("- ").strip() for idea in ideas_text.split('\n') if idea]
        return ideas

    except Exception as e:
        logging.error(f"Error generating ideas: {e}")
        return []
