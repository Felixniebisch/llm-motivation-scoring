import os
import sys
import json

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

from PydanticClasses import (
    InterestEnjoyment,
    PerceivedCompetence,
    EffortImportance,
    PressureTension,
    PerceivedChoice,
    ValueUsefulness,
)
from reasoning_prompts import reasoning_prompts
from reversed_scales import reversed_scales
from initialization_prompt import initialization_prompt

load_dotenv()  # loads environment variables from the .env file

# Point these at real files (env vars or edit the defaults) before running.
INPUT_CSV = os.environ.get("INPUT_CSV", "data/input.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "data/results.csv")
MODEL = "gpt-4o-2024-08-06"

# One entry per subscale: (key used in reasoning_prompts/reversed_scales,
# Pydantic response model mirroring the questionnaire, number of items,
# output column label, inner per-row dict key). The label and inner key
# are kept separate on purpose: the original script used two different
# strings for several subscales (e.g. outer "interest enjoyment" vs. inner
# "Interest_enjoyment"), and `compute averages.py` parses rows by looking
# for that exact inner key — changing it would silently break that script.
SUBSCALES = [
    ("Interest_enjoyment", InterestEnjoyment, 7, "interest enjoyment", "Interest_enjoyment"),
    ("Perceived_competence", PerceivedCompetence, 6, "perceived competence", "Perceived competence"),
    ("Effort_importance", EffortImportance, 6, "Effort importance", "Effort importance"),
    ("Pressure_tension", PressureTension, 5, "Pressure Tension", "Pressure tension"),
    ("Perceived_choice", PerceivedChoice, 7, "Perceived choice", "Perceived choice"),
    ("Value_usefulness", ValueUsefulness, 4, "Value Usefulness", "Value Usefulness"),
]


def reverse_scale_answers(answers, subscale_name, reversed_scales):
    """Reverse-code the configured items for `subscale_name`, in place."""
    for q_idx in reversed_scales.get(subscale_name, []):
        question_key = f"question{q_idx}"
        if question_key not in answers:
            print(f"No entry found for {question_key} in answers.")
            continue
        try:
            original_value = int(answers[question_key])
        except (TypeError, ValueError):
            print(f"Non-integer value for {question_key}: {answers[question_key]}")
            continue
        if 1 <= original_value <= 7:
            adjusted_value = 8 - original_value
            answers[question_key] = adjusted_value
            print(f"Reversed scale: {subscale_name}, {question_key} {original_value} -> {adjusted_value}")
        else:
            print(f"Value out of range for reversal: {original_value} in {question_key}")


def score_subscale(client, row_data, subscale_key, response_model, n_questions):
    """Ask the model to score one subscale for one participant's free text."""
    chat_completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "user", "content": " ".join(map(str, row_data))},
            {
                "role": "system",
                "content": (
                    "Please answer the following questions in JSON format with an "
                    "integer between 1 and 7 and explain your reasoning in regards "
                    "to the user input provided:\n" + "\n".join(reasoning_prompts[subscale_key])
                ),
            },
        ],
        response_format=response_model,
    )
    if not (chat_completion and chat_completion.choices):
        raise ValueError(f"No response for subscale {subscale_key}")

    # NOTE: `.parse()` already returns a validated object at `.message.parsed`.
    # Re-parsing `.content` as raw JSON (kept here to match the original
    # behavior) throws that validation away — worth switching to `.parsed`
    # once you've checked it against your installed openai SDK version.
    response_json = json.loads(chat_completion.choices[0].message.content)
    questions = [f"question{i}" for i in range(1, n_questions + 1)]
    answers = {key: response_json.get(key) for key in questions}
    reverse_scale_answers(answers, subscale_key, reversed_scales)
    return answers


def process_data(df):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    try:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": initialization_prompt}],
        )
        print("Initialization successful!\n")
    except Exception as e:
        sys.exit(f"Error in initialization prompt: {e}")

    results = []
    for index, row in df.iterrows():
        row_data = row.iloc[1:].tolist()
        subscale_results = {"Index": index}

        for subscale_key, response_model, n_questions, column_label, inner_key in SUBSCALES:
            try:
                answers = score_subscale(client, row_data, subscale_key, response_model, n_questions)
                subscale_results[column_label] = [{"Index": index, inner_key: answers}]
                print(f"Row {index}, {subscale_key}: {answers}")
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Row {index}, {subscale_key}: could not parse/validate response ({e})")
                subscale_results[column_label] = []
            except Exception as e:
                print(f"Row {index}, {subscale_key}: unexpected error ({e})")
                subscale_results[column_label] = []

        results.append(subscale_results)

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, delimiter=",", quotechar='"')
    processed_df = process_data(df)
    processed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results written to {OUTPUT_CSV}")
