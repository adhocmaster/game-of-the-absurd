import random
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
import re

game_symbols = ["player", "ball", "net"]
action_symbols = ["hits", "misses"]
comparator_symbols = ["most", "least"]
score_symbols = ["score", "points", "cars", "ice creams"]

free_models = ['deepseek/deepseek-chat-v3-0324:free',
               'google/gemini-2.0-flash-exp:free',
               'mistralai/mistral-nemo:free',
               'google/gemma-3-27b-it:free',
               'meta-llama/llama-4-maverick:free',
               'mistralai/mistral-small-3.1-24b-instruct:free',
               'qwen/qwen2.5-vl-72b-instruct:free',
               'mistralai/devstral-small:free',
               'mistralai/mistral-small-3.2-24b-instruct:free',
               'moonshotai/kimi-dev-72b:free']

cheap_models = ['google/gemini-2.0-flash-001',
                'openai/gpt-4o-mini',
                'meta-llama/llama-4-maverick-17b-128e-instruct',
                'qwen/qwen-2.5-72b-instruct',
                'google/gemini-2.5-flash']

expensive_models = ['anthropic/claude-3-5-haiku',
                    'nousresearch/hermes-3-llama-3.1-405b',
                    'sao10k/l3.1-euryale-70b',
                    'meta-llama/llama-3.1-405b-instruct',
                    'thedrummer/skyfall-36b-v2',
                    'openai/gpt-4.1-mini',
                    'google/gemma-2-27b-it']

expensive_reasoning_models = ['deepseek/deepseek-r1-0528',
                              'thedrummer/valkyrie-49b-v1',
                              'mistralai/magistral-small-2506',
                              'nvidia/llama-3.1-nemotron-ultra-253b-v1',
                              'perplexity/sonar-reasoning']

results = ['team a', 'team b', 'both']

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
        winner = "both"

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
    prompt = f"Absurd soccer is played by two teams of {game_symbols[game_state[0]]}s. Each team starts out with zero {score_symbols[score_state]}. In one match of this game, each team takes a turn to shoot a {game_symbols[game_state[1]]} five times at a {game_symbols[game_state[2]]}. A team can shoot only once in a match. When one team shoots, the other team defends the {game_symbols[game_state[2]]}. If the team that makes the shot {action_symbols[action_state]} the {game_symbols[game_state[2]]}, their team's {score_symbols[score_state]} increases by 1. At the end of the match, the team having the {comparator_symbols[comparator_state]} {score_symbols[score_state]} wins.\n\n"
    prompt += "Here is the match commentary for a game of absurd soccer:\n\n"
    game, answer = generate_game(game_state, action_state, comparator_state)
    prompt += game
    prompt += "\nWho won the game? Answer 'team A' if team A wins, 'team B' if team B wins, and 'both' if both teams wins. Please place your answer within two curly brackets (ex. {team A})."
    return prompt, answer

def task_1(api_key: str, num_sims: int, game_state: list, action_state: int, comparator_state: int, score_state: int, model_names):
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

    if model_names == "cheap":
        models = cheap_models
    elif model_names == "expensive":
        models = expensive_models
    elif model_names == "free":
        models = free_models
    elif model_names == "reasoning":
        models = expensive_reasoning_models
    elif type(model_names) is list:
        models = model_names
    else:
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    data = {
        'game #': [],
        'prompt': [],
        'answer': [],
    }
    total_results = {}
    for model in models:
        data[model] = []
        total_results[model] = 0
    for i in range(num_sims):
        prompt, answer = generate_prompt_1(game_state, action_state, comparator_state, score_state)
        data['game #'].append(i)
        data['prompt'].append(prompt)
        data['answer'].append(answer)
        
        for model in models:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            while completion.choices == None or (completion.choices[0].message.content != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()).lower() not in results):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            data[model].append(re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()))
            if data['answer'][-1].lower() == data[model][-1].lower():
                total_results[model] += 1
        
        print("Generated game", str(i))

    data['game #'].append('total')
    data['prompt'].append(None)
    data['answer'].append(None)
    for model in models:
        data[model].append(total_results[model] / num_sims)

    df = pd.DataFrame(data)
    return df

def generate_empty_game(game_state: list, comparator_state: int):
    """
    Generates incomplete match commentary for a game of absurd soccer, where each team has 5 turns. Commentary is missing values for actions.
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    """
    game = ""
    for i in range(5):
        game += f"Team A shoots the {game_symbols[game_state[1]]} and {{}} the {game_symbols[game_state[2]]}.\n"
        game += f"Team B shoots the {game_symbols[game_state[1]]} and {{}} the {game_symbols[game_state[2]]}.\n"
    
    return game

def generate_prompt_2(game_state: list, action_state: int, comparator_state: int, score_state: int, outcome: str):
    """
    Generates prompts for models which describe the rules of absurd soccer, provides incomplete commentary and outcome for a game, and asks to complete the commentary.
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - outcome: outcome of the game
    """
    prompt = ""
    prompt = f"Absurd soccer is played by two teams of {game_symbols[game_state[0]]}s. Each team starts out with zero {score_symbols[score_state]}. In one match of this game, each team takes a turn to shoot a {game_symbols[game_state[1]]} five times at a {game_symbols[game_state[2]]}. A team can shoot only once in a match. When one team shoots, the other team defends the {game_symbols[game_state[2]]}. If the team that makes the shot {action_symbols[action_state]} the {game_symbols[game_state[2]]}, their team's {score_symbols[score_state]} increases by 1. At the end of the match, the team having the {comparator_symbols[comparator_state]} {score_symbols[score_state]} wins.\n\n"
    prompt += "Here is an incomplete match commentary for a game of absurd soccer:\n\n"
    game = generate_empty_game(game_state, action_state)
    prompt += game
    prompt += f"\nThe outcome of the game is that {outcome} wins. Your task is to complete the rest of the commentary by filling in the missing values (denoted by brackets {{}}) with either 'hits' or 'misses' such that it matches the rules and the outcomes. You will do this by generating a list of 10 words, with each word either being 'hits' or 'misses', such that the order of the words correspond to the order of the missing values in the game commentary. Please format the list within brackets (ex. {{hits,misses,hits,misses,misses}})"
    return prompt

def task_2(api_key: str, num_sims: int, game_state: list, action_state: int, comparator_state: int, score_state: int, model_names):
    """
    Tests each model's ability to complete commentary for a game of absurd soccer using the generate_prompt_2 function
    - api_key: OpenRouter API Key
    - num_sims: number of simulations
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - model_names: either a list of model names from OpenRouter, or a string denoting a curated set of models
        - "free": free models
        - "cheap": $0.1-$0.2 per token
        - "expensive": $0.5-$1 per token
        - "reasoning": reasoning models that are $0.5-$1 per token
    """
    if model_names == "cheap":
        models = cheap_models
    elif model_names == "expensive":
        models = expensive_models
    elif model_names == "free":
        models = free_models
    elif model_names == "reasoning":
        models = expensive_reasoning_models
    elif type(model_names) is list:
        models = model_names
    else:
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    data = {
        'game #': [],
        'prompt': [],
        'answer': [],
    }
    total_results = {}
    for model in models:
        data[model + '_response'] = []
        data[model + '_values'] = []
        data[model + '_outcome'] = []
        total_results[model] = 0
    
    for i in range(num_sims):
        outcome = random.choice(['team A', 'team B', 'both teams'])
        prompt = generate_prompt_2(game_state, action_state, comparator_state, score_state, outcome)
        data['game #'].append(i)
        data['prompt'].append(prompt)
        data['answer'].append(outcome)
        
        for model in models:
            print("Testing model:", model)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            missing_values = re.sub(r'[^a-zA-Z0-9,]+', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip().lower()).split(",")
            while completion.choices == None or len(missing_values) != 10 or not all(word in ("hits", "misses") for word in missing_values):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                missing_values = re.sub(r'[^a-zA-Z0-9,]+', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip().lower()).split(",")
            data[model + '_response'].append(completion.choices[0].message.content)
            data[model + '_values'].append(missing_values)
            A_score = 0
            B_score = 0
            for j in range(len(missing_values)):
                if missing_values[j] == action_symbols[action_state]:
                    if j % 2 == 0:
                        A_score += 1
                    if j % 2 == 1:
                        B_score += 1
            
            if (A_score > B_score and comparator_state == 0) or (A_score < B_score and comparator_state == 1):
                data[model + '_outcome'].append('team A')
            elif (A_score < B_score and comparator_state == 0) or (A_score > B_score and comparator_state == 1):
                data[model + '_outcome'].append('team B')
            else:
                data[model + '_outcome'].append('both teams')
            
            if data['answer'][-1].lower() == data[model + '_outcome'][-1].lower():
                total_results[model] += 1
            
        
        print("Generated game", str(i))

    data['game #'].append('total')
    data['prompt'].append(None)
    data['answer'].append(None)
    for model in models:
        data[model + '_values'].append(None)
        data[model + '_response'].append(None)
        data[model + '_outcome'].append(total_results[model] / num_sims)

    df = pd.DataFrame(data)
    return df

def generate_prompt_2_alternate(game_state: list, action_state: int, comparator_state: int, score_state: int, outcome: str):
    """
    Generates prompts for models which describe the rules of absurd soccer and provide example commentary. Describes an outcome for a game, and asks to write the commentary.
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - outcome: outcome of the game
    """
    prompt = ""
    prompt = f"Absurd soccer is played by two teams of {game_symbols[game_state[0]]}s. Each team starts out with zero {score_symbols[score_state]}. In one match of this game, each team takes a turn to shoot a {game_symbols[game_state[1]]} five times at a {game_symbols[game_state[2]]}. A team can shoot only once in a match. When one team shoots, the other team defends the {game_symbols[game_state[2]]}. If the team that makes the shot {action_symbols[action_state]} the {game_symbols[game_state[2]]}, their team's {score_symbols[score_state]} increases by 1. At the end of the match, the team having the {comparator_symbols[comparator_state]} {score_symbols[score_state]} wins.\n\n"
    prompt += "Here is an example of the match commentary for a game of absurd soccer:\n\n"
    game, answer = generate_game(game_state, action_state, comparator_state)
    prompt += game
    prompt += f"\nYour task is to generate match commentary for a game of absurd soccer such that {outcome} wins. The commentary should be in the same format as the commentary above, and should adhere to the ruleset for absurd soccer. In your response, please insert the match commentary within brackets. (ex. {{match comentary goes here}})"
    return prompt

def task_2_alternate(api_key: str, num_sims: int, game_state: list, action_state: int, comparator_state: int, score_state: int, model_names):
    """
    Tests each model's ability to write commentary for a game of absurd soccer using the generate_prompt_2 function
    - api_key: OpenRouter API Key
    - num_sims: number of simulations
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - model_names: either a list of model names from OpenRouter, or a string denoting a curated set of models
        - "free": free models
        - "cheap": $0.1-$0.2 per token
        - "expensive": $0.5-$1 per token
        - "reasoning": reasoning models that are $0.5-$1 per token
    """
    if model_names == "cheap":
        models = cheap_models
    elif model_names == "expensive":
        models = expensive_models
    elif model_names == "free":
        models = free_models
    elif model_names == "reasoning":
        models = expensive_reasoning_models
    elif type(model_names) is list:
        models = model_names
    else:
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    data = {
        'game #': [],
        'prompt': [],
        'answer': [],
    }
    for model in models:
        data[model + '_response'] = []
        data[model + '_commentary'] = []
        # OUTCOME CELLS NEED TO BE FILLED IN MANUALLY
        data[model + '_outcome'] = []
    
    for i in range(num_sims):
        outcome = random.choice(['team A', 'team B', 'both teams'])
        prompt = generate_prompt_2_alternate(game_state, action_state, comparator_state, score_state, outcome)
        data['game #'].append(i)
        data['prompt'].append(prompt)
        data['answer'].append(outcome)
        
        for model in models:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            while completion.choices == None:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            commentary = completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()
            data[model + '_response'].append(completion.choices[0].message.content)
            data[model + '_commentary'].append(commentary)
            lines = commentary.split("\n")

            #first_line = -1
            
            #print("len(lines)", len(lines))
            for n, line in enumerate(lines):
                if "Team A shoots the" in line:
                    first_line = n
                    break
            

            if first_line < 0 or len(lines) < first_line + 10:
                data[model + '_outcome'].append(None)
                continue
            
            A_score = 0
            B_score = 0

            valid = True
    
            for j in range(10):
                team = "A" if j % 2 == 0 else "B"
                words = lines[first_line + j].split(' ')
                #print(words)
                if f"Team {team} shoots the" in lines[first_line + j]:
                    if words[6] == "misses" and action_state == 1:
                        if team == "A":
                            A_score += 1
                        if team == "B":
                            B_score += 1
                    elif words[6] == "hits" and action_state == 0:
                        if team == "A":
                            A_score += 1
                        if team == "B":
                            B_score += 1
                    elif words[6] != "hits" and words[6] != "misses":
                        data[model + '_outcome'].append(None)
                        valid = False
                        break
                else:
                    data[model + '_outcome'].append(None)
                    valid = False
                    break
            
            if not valid:
                break
            
            if (A_score > B_score and comparator_state == 0) or (A_score < B_score and comparator_state == 1):
                data[model + '_outcome'].append('team A')
            elif (A_score < B_score and comparator_state == 0) or (A_score > B_score and comparator_state == 1):
                data[model + '_outcome'].append('team B')
            else:
                data[model + '_outcome'].append('both teams')
        
        print(model)
        print(len(data[model + '_commentary']))
        print(len(data[model + '_response']))
        print(len(data[model + '_outcome']))
        print("Generated game", str(i))   
        

    data['game #'].append('total')
    #print(len(data['game #']))
    data['prompt'].append(None)
    #print(len(data['prompt']))
    data['answer'].append(None)
    #print(len(data['answer']))
    for model in models:
        print(model)
        data[model + '_commentary'].append(None)
        print(len(data[model + '_commentary']))
        data[model + '_response'].append(None)
        print(len(data[model + '_response']))
        data[model + '_outcome'].append(None)
        print(len(data[model + '_outcome']))

    df = pd.DataFrame(data)
    return df

def save_results_to_file(df, identifier):
    """
    Saves results dataframe to a csv file
    - df: DataFrame that should be saved
    - identifier: keyword or identifier that file should be named with
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    df.to_csv(identifier + '_results' + today_str + '.csv', index=False)
