import random
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
import re
import statistics
import math
import requests
import json
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig

game_symbols = ["player", "ball", "net"]
action_symbols = ["hits", "misses"]
comparator_symbols = ["most", "least"]
score_symbols = ["score", "points", "cars", "ice creams"]

openai_models = ["openai/gpt-4o-mini", # CHEAP
                 "openai/gpt-4o-mini-2024-07-18", # CHEAP
                 "openai/gpt-3.5-turbo"] # EXPENSIVE

gemini_models = ["gemini-2.5-flash-lite", # CHEAP
                 "gemini-2.0-flash-001", # CHEAP
                 "gemini-3-flash-preview"] # EXPENSIVE

results = ['team a', 'team b', 'both teams']

worst_prompts = pd.read_csv("worst_prompts_alternate_task_2.csv")

def calc_openai_model_entropy(model_response):
    response_entropy = 0
    for token in model_response.json()['choices'][0]['logprobs']['content']:
        token_entropy = 0
        for possible_token in token['top_logprobs']:
            logp = possible_token['logprob']
            token_entropy -= logp * math.exp(logp)
        response_entropy += token_entropy
    
    response_entropy /= len(model_response.json()['choices'][0]['logprobs']['content'])
    return response_entropy

def openai_response(model, message, key):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "logprobs": True,
            "top_logprobs": 20
        })
    )
    # text_response = response.json()['choices'][0]['message']['content']
    # entropy = calc_openai_model_entropy(response)
    return response

def calc_gemini_model_entropy(response):
    response_entropy = 0
    for token in response.candidates[0].logprobs_result.top_candidates:
        token_entropy = 0
        for possible_token in token.candidates:
            logp = possible_token.log_probability
            token_entropy -= logp * math.exp(logp)
        response_entropy += token_entropy
    
    response_entropy /= len(response.candidates[0].logprobs_result.top_candidates)
    return response_entropy

def gemini_response(model, message, project_id):
    client = genai.Client(vertexai=True, project=project_id, location="global", http_options=HttpOptions(api_version="v1"))
    response = client.models.generate_content(
        model=model,
        contents=message,
        config=GenerateContentConfig(
            response_logprobs=True,  # turn logprobs output on
            logprobs=20,              # top 3 alternatives at each step
        )
    )

    # text_response = response.candidates[0].content.parts[0].text
    # entropy = calc_gemini_model_entropy(response)
    return response

def save_results_to_file(df, identifier):
    """
    Saves results dataframe to a csv file
    - df: DataFrame that should be saved
    - identifier: keyword or identifier that file should be named with
    """

    try:
        os.remove("./"+identifier + '.csv')
    except:
        pass
    df.to_csv(identifier + '.csv', index=False)

def turn_ruleset_to_settings(ruleset: str):
    game_state = [0, 1, 2]
    action_state = 0
    comparator_state = 0
    score_state = 0

    if ruleset not in ["Default", "Switch", "Miss Switch", "Miss", "Less", "Car", "Ice Cream"]:
        quit()

    if ruleset == "Switch" or ruleset=="Miss Switch":
        game_state = [0, 2, 1]
    if ruleset == "Miss" or ruleset=="Miss Switch":
        action_state = 1
    if ruleset == "Less":
        comparator_state = 1
    if ruleset == "Car":
        score_state = 2
    if ruleset == "Ice Cream":
        score_state = 3
    
    return game_state, action_state, comparator_state, score_state

def generate_game(game_state: list, action_state: int, comparator_state: int):
    """
    Generates match commentary for a game of absurd soccer, where each team has 5 turns
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    """
    game = ""
    A = []
    B = []

    for i in range(5):
        A.append(random.randint(0, 1))
        B.append(random.randint(0, 1))
        game += f"Team A shoots the {game_symbols[game_state[1]]} and {action_symbols[action_state if A[i] else 1-action_state]} the {game_symbols[game_state[2]]}.\n"
        game += f"Team B shoots the {game_symbols[game_state[1]]} and {action_symbols[action_state if B[i] else 1-action_state]} the {game_symbols[game_state[2]]}.\n"

    if A.count(1) > B.count(1):
        winner = "team A" if comparator_state == 0 else "team B"
    elif A.count(1) < B.count(1):
        winner = "team B" if comparator_state == 0 else "team A"
    else:
        winner = "both teams"

    return game, winner

def generate_prompt_1(game_state: list, action_state: int, comparator_state: int, score_state: int):
    """
    Generates prompts for models which describe the rules of absurd soccer, provide commentary for a game, and asks for the outcome of that game
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    """
    prompt = ""
    prompt = f"Absurd soccer is played by two teams of {game_symbols[game_state[0]]}s. Each team starts out with zero {score_symbols[score_state]}. The game consists of five matches; in each match, both teams shoot a {game_symbols[game_state[1]]} at a {game_symbols[game_state[2]]}. When one team shoots, the other team defends the {game_symbols[game_state[2]]}. If the team that makes the shot {action_symbols[action_state]} the {game_symbols[game_state[2]]}, their team's {score_symbols[score_state]} increases by 1. At the end of the game, the team having the {comparator_symbols[comparator_state]} {score_symbols[score_state]} wins.\n\n"
    prompt += "Here is the match commentary for a game of absurd soccer:\n\n"
    game, answer = generate_game(game_state, action_state, comparator_state)
    prompt += game
    prompt += "\nWho won the game? Answer 'team A' if team A wins, 'team B' if team B wins, and 'both teams' if both teams wins. Please work out your reasoning process for the answer, and place your answer within two curly brackets (ex. {team A})."
    return prompt, answer

def task_1(openai_api_key: str, gemini_project_id: str, num_sims: int, ruleset: str, file_name: str):
    """
    Tests each model's ability to evaluate a game of absurd soccer using the generate_prompt_1 function
    - api_key: OpenRouter API Key
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - models: either a list of model names from OpenRouter, or a string denoting a curated set of models
        - "free": free models
        - "cheap": $0.1-$0.2 per token
        - "expensive": $0.5-$1 per token
        - "reasoning": reasoning models that are $0.5-$1 per token
    """

    game_state, action_state, comparator_state, score_state = turn_ruleset_to_settings(ruleset)
    models = openai_models + gemini_models

    try:
        df = pd.read_csv(file_name+'.csv')
    except:
        data = {
            'game #': [],
            'prompt': [],
            'answer': [],
        }

        for model in models:
            data[model + '_response'] = []
            data[model + '_outcome'] = []
            data[model + '_entropy'] = []
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return

    for i in range(num_sims-original_length):
        new_row = {}
        prompt, answer = generate_prompt_1(game_state, action_state, comparator_state, score_state)
        new_row['game #'] = [i+original_length]
        new_row['prompt'] = [prompt]
        new_row['answer'] = [answer]
        
        print("Generating game", str(i+original_length))  
        
        for model in models:
            print("Testing model:", model)
            if model.startswith("openai"):
                completion = openai_response(model, prompt, openai_api_key)
                while completion.json()['choices'] == None or (completion.json()['choices'][0]['message']['content'] != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.json()['choices'][0]['message']['content'].split("{")[-1].split("}")[0].strip()).lower() not in results):
                    completion = openai_response(model, prompt, openai_api_key)
                new_row[model + '_response'] = [completion.json()['choices'][0]['message']['content']]
                new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', new_row[model + '_response'].split("{")[-1].split("}")[0].strip())]
                new_row[model + '_entropy'] = [calc_openai_model_entropy(completion)]

            else:
                completion = gemini_response(model, prompt, gemini_project_id)
                while re.sub(r'[^a-zA-Z0-9 ]', '', completion.candidates[0].content.parts[0].text.split("{")[-1].split("}")[0].strip()).lower() not in results:
                    completion = gemini_response(model, prompt, gemini_project_id)
                new_row[model + '_response'] = [completion.candidates[0].content.parts[0].text]
                new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', new_row[model + '_response'].split("{")[-1].split("}")[0].strip())]
                new_row[model + '_entropy'] = [calc_gemini_model_entropy(completion)]
            
        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])
        save_results_to_file(df, file_name)

    total_results = {}
    new_row = {}
    new_row['game #'] = ['total']
    new_row['prompt'] = [None]
    new_row['answer'] = [None]
    for model in models:
        total_results[model] = 0
        avg_entropy = 0
        for i in range(num_sims):
            if type(list(df[model + '_outcome'])[i]) == str:
                if list(df['answer'])[i].lower() == list(df[model + '_outcome'])[i].lower():
                    total_results[model] += 1
            avg_entropy += list(df[model+'_entropy'])[i]

        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]
        new_row[model + '_entropy'] = [avg_entropy / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)

def generate_prompt_1_few_shot(ruleset:str, prompts):
    game_state, action_state, comparator_state, score_state = turn_ruleset_to_settings(ruleset)
    sample_prompts = [prompts[ruleset][i] for i in range(len(prompts[ruleset])) if type(prompts[ruleset][i]) is str and prompts[ruleset][i].startswith("Absurd")]
    sample_answers = [prompts[ruleset+"_answer"][i] for i in range(len(prompts[ruleset])) if type(prompts[ruleset][i]) is str and prompts[ruleset][i].startswith("Absurd")]
    #print(sample_answers)
    random_index = random.sample(range(0, len(sample_prompts)), 4)
    prompt = ""
    for i, index in enumerate(random_index):
        prompt += "Question:\n"
        prompt += str(sample_prompts[index])
        prompt += "\n\n"
        prompt += "Answer:\n"
        if i != 3:
            prompt += "{" + str(sample_answers[index]) + "}\n\n"

    return prompt, sample_answers[random_index[-1]]

def task_1_few_shot(openai_api_key: str, gemini_project_id: str, num_sims: int, ruleset: str, file_name: str):
    """
    Tests each model's ability to evaluate a game of absurd soccer using the generate_prompt_1 function
    - api_key: OpenRouter API Key
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - models: either a list of model names from OpenRouter, or a string denoting a curated set of models
        - "free": free models
        - "cheap": $0.1-$0.2 per token
        - "expensive": $0.5-$1 per token
        - "reasoning": reasoning models that are $0.5-$1 per token
    """

    game_state, action_state, comparator_state, score_state = turn_ruleset_to_settings(ruleset)

    models = openai_models + gemini_models

    try:
        df = pd.read_csv(file_name+'.csv')
    except:
        data = {
            'game #': [],
            'prompt': [],
            'answer': [],
        }

        for model in models:
            data[model + '_response'] = []
            data[model + '_outcome'] = []
            data[model + '_entropy'] = []
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return

    for i in range(num_sims-original_length):
        new_row = {}
        bad_prompts = pd.read_csv("worst_prompts_DO_"+ruleset+".csv")
        prompt, answer = generate_prompt_1_few_shot(ruleset, bad_prompts)
        new_row['game #'] = [i+original_length]
        new_row['prompt'] = [prompt]
        new_row['answer'] = [answer]
        print("Generating game", str(i+original_length))  
        
        for model in models:
            print("Testing model:", model)
            if model.startswith("openai"):
                completion = openai_response(model, prompt, openai_api_key)
                while completion.json()['choices'] == None or (completion.json()['choices'][0]['message']['content'] != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.json()['choices'][0]['message']['content'].split("{")[-1].split("}")[0].strip()).lower() not in results):
                    completion = openai_response(model, prompt, openai_api_key)
                new_row[model + '_response'] = [completion.json()['choices'][0]['message']['content']]
                new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', new_row[model + '_response'].split("{")[-1].split("}")[0].strip())]
                new_row[model + '_entropy'] = [calc_openai_model_entropy(completion)]

            else:
                completion = gemini_response(model, prompt, gemini_project_id)
                while re.sub(r'[^a-zA-Z0-9 ]', '', completion.candidates[0].content.parts[0].text.split("{")[-1].split("}")[0].strip()).lower() not in results:
                    completion = gemini_response(model, prompt, gemini_project_id)
                new_row[model + '_response'] = [completion.candidates[0].content.parts[0].text]
                new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', new_row[model + '_response'].split("{")[-1].split("}")[0].strip())]
                new_row[model + '_entropy'] = [calc_gemini_model_entropy(completion)]

        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])
        save_results_to_file(df, file_name)

    total_results = {}
    new_row = {}
    new_row['game #'] = ['total']
    new_row['prompt'] = [None]
    new_row['answer'] = [None]
    for model in models:
        avg_entropy = 0
        total_results[model] = 0
        for i in range(num_sims):
            if type(list(df[model + '_outcome'])[i]) == str:
                if list(df['answer'])[i].lower() == list(df[model + '_outcome'])[i].lower():
                    total_results[model] += 1
            avg_entropy += list(df[model+'_entropy'])[i]

        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]
        new_row[model + '_entropy'] = [avg_entropy / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)

def run_all_models(task: str, openai_api_key: str, gemini_project_id: str, num_sims: int, ruleset: str):
    if task == "DO":
        for i in range(num_sims):
            task_1(openai_api_key, gemini_project_id, num_sims, ruleset, "do_"+ruleset)
    elif task == "DOFS":
        for i in range(num_sims):
            task_1_few_shot(openai_api_key, gemini_project_id, num_sims, ruleset, "dofs_"+ruleset)
  
def run_all_rulesets(task: str, openai_api_key: str, gemini_project_id: str, num_sims: int):
    all_rulesets = ["Default", "Switch", "Miss Switch", "Miss", "Less", "Car", "Ice Cream"]

    if not os.path.exists(task):
        os.mkdir(task)
    
    os.chdir(task)

    for r in all_rulesets:
        run_all_models(task, openai_api_key, gemini_project_id, num_sims, r)
    
    os.chdir('..')
    
def get_worse_results(tasks: list):
    rulesets = ["Car", "Default", "Ice Cream", "Less", "Miss", "Miss Switch", "Switch"]
    for t in tasks:
        os.chdir(t)
        for r in rulesets:
            worst_prompts = {r: [], r+"_answer": []}
            results = pd.read_csv(t.lower() + "_" + r + ".csv")
            scores = []
            for i in range(len(results)-1):
                count = 0
                for m in openai_models + gemini_models:
                    if results[m+"_outcome"][i] == results["answer"][i]:
                        count += 1
                scores.append(count)
            for i in range(len(results)-1):
                if scores[i] < statistics.median(scores):
                    worst_prompts[r].append(results["prompt"][i])
                    worst_prompts[r+"_answer"].append(results["answer"][i])
            worst_prompts = pd.DataFrame(worst_prompts)
            worst_prompts.to_csv("worst_prompts_" + t + "_" + r + ".csv")
        os.chdir('..')

def run_full_exp(folder_name, openai_api_key, gemini_project_id, num_sims):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)
    all_tasks = ["DO", "DOFS"]
    for t in all_tasks:
        run_all_rulesets(t, openai_api_key, gemini_project_id, num_sims)
        if t == "DO":
            get_worse_results(t)
            



