# idea_generation.py

import openai
import os
import logging

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_ideas():
    prompt = (
        "Generate five novel research ideas in the field of artificial intelligence (AI) and machine learning (ML) "
        "that focus on improving the performance and capabilities of AI systems themselves. "
        "The ideas should be executable with limited computational resources (e.g., single GPU) and within one week. "
        "Ensure that the ideas are specific to AI/ML and aim to enhance the AI Research System's own performance. "
        "List them as bullet points."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use a valid model name, e.g., "gpt-3.5-turbo" or "gpt-4"
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates AI research ideas."},
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
