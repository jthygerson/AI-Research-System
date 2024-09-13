# idea_generation.py

import openai
import os
import logging

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_ideas():
    prompt = (
        "Generate five novel research ideas in the field of AI that can be executed "
        "with limited computational resources (e.g., single GPU) and within one week. "
        "List them as bullet points."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Changed model to 'gpt-4o'
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates research ideas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            temperature=0.7,
            stop=["\n\n"]  # Stop after generating the list
        )

        ideas_text = response['choices'][0]['message']['content'].strip()
        # Split the ideas into a list
        ideas = [idea.strip('-• ').strip() for idea in ideas_text.split('\n') if idea.strip()]
        logging.info(f"Generated ideas: {ideas}")
        return ideas

    except Exception as e:
        logging.error(f"Error generating ideas: {e}")
        return []
