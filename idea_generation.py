# idea_generation.py

import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def generate_ideas():
    prompt = (
        "Generate five novel research ideas in the field of AI that can be executed "
        "with limited computational resources (e.g., single GPU) and within one week."
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    ideas = response.choices[0].text.strip().split('\n')
    return [idea for idea in ideas if idea]

