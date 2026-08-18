import os
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import  BaseModel, ValidationError
from typing import List
import pandas as pd
import sys
import json
from typing import Dict
from PydanticClasses import interestEnjoyment, PerceivedCompetence, EffortImportance, PressureTension,  PerceivedChoice, ValueUsefulness
from reasoning_prompts import reasoning_prompts
from reversed_scales import reversed_scales
from initialization_prompt import initialization_prompt

# pydantic classes that serve the output structuring
# each class mirrors the amount of questions in the original questionnaire and thereby forces gpt to answer every one of them


load_dotenv()  #  loads the environment variables from the .env file

def reverse_scale_answers(answers, subscale_name, reversed_scales):
   
    reversed_indices = reversed_scales.get(subscale_name, [])
    
    for q_idx in reversed_indices:
        question_key = f'question{q_idx}'
        
        if question_key in answers:
            try:
                original_value = int(answers[question_key])
                
                if 1 <= original_value <= 7:
                    adjusted_value = 8 - original_value
                    answers[question_key] = adjusted_value
                    print(f"Reversed Scale: {subscale_name}, {question_key} adjusted from {original_value} to {adjusted_value}")
                else:
                    print(f"Value out of range for reversal: {original_value} in {question_key}")
            except ValueError:
                print(f"Non-integer value for {question_key}: {answers[question_key]}")
        else:
            print(f"No entry found for {question_key} in answers.")


def process_data(df):
    results = []
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    print(df.head())  # To check initial data
    print(df.isna().sum())
    
    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini", #adjust model if wanted
        messages=[
            {"role": "system", "content": initialization_prompt}
        ]
    )
        print('Initialization successful!\n')
    except Exception as e:
        print(f"Error in initialization prompt: {e}")
        sys.exit("Terminating program.")

    for index, row in df.iterrows():
        row_data = row.iloc[1:].tolist()
        print(f'Row {index} data:', row_data)
        subscale_results = {'Index': index}

        try: 
                    Interest_enjoyment_results = []
                    chat_completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                             {"role": "user", "content": " ".join(map(str, row_data))},
                             {"role" : "system", "content": "Please answer the following questions in JSON format with in integer between 1 and 7 and explain your reasoning in regards to the user input provided:\n" + "\n".join(reasoning_prompts["Interest_enjoyment"])}
                             ],
                        response_format = interestEnjoyment
                    )
                    print(chat_completion)

                    if chat_completion and chat_completion.choices:
                        formatted_response = chat_completion.choices[0].message.content
                    else:
                        formatted_response = "No response"
                    
                    response_json = json.loads(formatted_response)
                    questions = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6', 'question7']
                    answers = {key: response_json.get(key, None) for key in questions}  # Using .get() to avoid KeyError

                    ######## ------ Handle reversed scales ------ #############
                    

                    # Append results to the list
                    interest_enjoyment_results.append({'Index' : index, 'Interest_enjoyment' : answers})
    
                    print(f"Row {index}: {formatted_response}")

                    reverse_scale_answers(answers, 'Interest_enjoyment', reversed_scales)

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {formatted_response}")
        except Exception as e:
            print(f"An error occurred: {e}")

        try: 
                    Perceived_competence_results = []
                    chat_completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                             {"role": "user", "content": " ".join(map(str, row_data))},
                             {"role" : "system", "content": "Please answer the following questions in JSON format with in integer between 1 and 7:\n" + "\n".join(reasoning_prompts["Perceived_competence"])}
                             ],
                        response_format = PerceivedCompetence
                    )

                    if chat_completion and chat_completion.choices:
                        formatted_response = chat_completion.choices[0].message.content
                    else:
                        formatted_response = "No response"
                    
                    response_json = json.loads(formatted_response)
                    questions = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6']
                    answers = {key: response_json.get(key, None) for key in questions}  # Using .get() to avoid KeyError


                    # Append results to the list
                    Perceived_competence_results.append({'Index' : index, 'Perceived competence' : answers})
    
                    print(f"Row {index}: {formatted_response}")

                    reverse_scale_answers(answers, 'Perceived_competence', reversed_scales)

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {formatted_response}")
        except Exception as e:
            print(f"An error occurred: {e}")

        try: 
                    Effort_importance_results = []
                    chat_completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                             {"role": "user", "content": " ".join(map(str, row_data))},
                             {"role" : "system", "content": "Please answer the following questions in JSON format with an integer between 1 and 7:\n" + "\n".join(reasoning_prompts["Effort_importance"])}
                             ],
                        response_format = EffortImportance
                    )

                    if chat_completion and chat_completion.choices:
                        formatted_response = chat_completion.choices[0].message.content
                    else:
                        formatted_response = "No response"
                    
                    response_json = json.loads(formatted_response)
                    questions = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6']
                    answers = {key: response_json.get(key, None) for key in questions}  # Using .get() to avoid KeyError
                    

                    # Append results to the list
                    Effort_importance_results.append({'Index' : index, 'Effort importance' : answers})
    
                    print(f"Row {index}: {formatted_response}")
                                        
                    reverse_scale_answers(answers, 'Effort_importance', reversed_scales)

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {formatted_response}")
        except Exception as e:
            print(f"An error occurred: {e}")

        try: 
                    Pressure_tension_results = []
                    chat_completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                             {"role": "user", "content": " ".join(map(str, row_data))},
                             {"role" : "system", "content": "Please answer the following questions in JSON format with in integer between 1 and 7:\n" + "\n".join(reasoning_prompts["Pressure_tension"])}
                             ],
                        response_format = PressureTension
                    )

                    if chat_completion and chat_completion.choices:
                        formatted_response = chat_completion.choices[0].message.content
                    else:
                        formatted_response = "No response"
                    
                    response_json = json.loads(formatted_response)
                    questions = ['question1', 'question2', 'question3', 'question4', 'question5']
                    answers = {key: response_json.get(key, None) for key in questions}  # Using .get() to avoid KeyError
                    

                    # Append results to the list
                    Pressure_tension_results.append({'Index' : index, 'Pressure tension' : answers})
    
                    print(f"Row {index}: {formatted_response}")

                    reverse_scale_answers(answers, 'Pressure_tension', reversed_scales)

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {formatted_response}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
        try: 
                    Perceived_choice_resuts = []
                    chat_completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                             {"role": "user", "content": " ".join(map(str, row_data))},
                             {"role" : "system", "content": "Please answer the following questions in JSON format with in integer between 1 and 7:\n" + "\n".join(reasoning_prompts["Perceived_choice"])}
                             ],
                        response_format = PerceivedChoice
                    )

                    if chat_completion and chat_completion.choices:
                        formatted_response = chat_completion.choices[0].message.content
                    else:
                        formatted_response = "No response"
                    
                    response_json = json.loads(formatted_response)
                    questions = ['question1', 'question2', 'question3', 'question4', 'question5', 'question6',  'question7']
                    answers = {key: response_json.get(key, None) for key in questions}  # Using .get() to avoid KeyError

                    

                    Perceived_choice_resuts.append({'Index' : index, 'Perceived choice' : answers})
    
                    print(f"Row {index}: {formatted_response}")

                    reverse_scale_answers(answers, 'Perceived_choice', reversed_scales)

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {formatted_response}")
        except Exception as e:
            print(f"An error occurred: {e}")

        try: 
                    Value_usefulness_results = []
                    chat_completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                             {"role": "user", "content": " ".join(map(str, row_data))},
                             {"role" : "system", "content": "Please answer the following questions in JSON format with in integer between 1 and 7:\n" + "\n".join(reasoning_prompts["Value_usefulness"])}
                             ],
                        response_format = ValueUsefulness
                    )

                    if chat_completion and chat_completion.choices:
                        formatted_response = chat_completion.choices[0].message.content
                    else:
                        formatted_response = "No response"
                    
                    response_json = json.loads(formatted_response)
                    questions = ['question1', 'question2', 'question3', 'question4']
                    answers = {key: response_json.get(key, None) for key in questions}  # Using .get() to avoid KeyError

                    # Append results to the list
                    Value_usefulness_results.append({'Index' : index, 'Value Usefulness' : answers})
    
                    print(f"Row {index}: {formatted_response}")

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {formatted_response}")
        except Exception as e:
            print(f"An error occurred: {e}")

        subscale_results['interest enjoyment'] = Interest_enjoyment_results
        subscale_results['perceived competence'] = Perceived_competence_results
        subscale_results['Effort importance'] =  Effort_importance_results
        subscale_results['Pressure Tension'] = Pressure_tension_results
        subscale_results['Perceived choice'] = Perceived_choice_resuts
        subscale_results['Value Usefulness'] = Value_usefulness_results

        results.append(subscale_results)



    return pd.DataFrame(results)


df = pd.read_csv('',  delimiter=',', quotechar='"')

# Process the data and get results
processed_df = process_data(df)

# Save processed results to CSV
processed_df.to_csv('', index=False) 
print("Results have been put out'")

