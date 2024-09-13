# idea_generation.py

import openai
import os
import logging

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_ideas():
    prompt = (
        "Generate five novel research ideas in the field of AI that can be executed "
        "with limited computational resources (e.g., single GPU) and within one week. "
        "List them as bullet points."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates research ideas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            n=1,
            temperature=0.7,
        )

        ideas_text = response['choices'][0]['message']['content'].strip()
        # Split ideas by newlines and bullets
        ideas = [idea.strip('-• ').strip() for idea in ideas_text.split('\n') if idea.strip()]
        logging.info(f"Generated ideas: {ideas}")
        return ideas

    except Exception as e:
        logging.error(f"Error generating ideas: {e}")
        return []
