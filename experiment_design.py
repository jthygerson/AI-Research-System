# experiment_design.py

import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def design_experiment(idea):
    prompt = (
        f"Design an experiment to test the following idea:\nIdea: {idea}\n"
        "Include the objective, methodology, datasets required, and evaluation metrics."
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=800,
        n=1,
        stop=None,
        temperature=0.7,
    )

    experiment_plan = response.choices[0].text.strip()
    return experiment_plan

