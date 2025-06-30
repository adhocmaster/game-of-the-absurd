import random
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
import re

game_symbols = ["player", "ball", "net"]
action_symbols = ["hits", "misses"]
comparator_symbols = ["most", "least"]
score_symbols = ["score", "point", "car", "ice-cream"]

free_models = ['deepseek/deepseek-chat-v3-0324:free',
               'deepseek/deepseek-r1-0528:free',
               'deepseek/deepseek-r1:free',
               'deepseek/deepseek-chat-v3:free',
               'tngtech/deepseek-r1t-chimera:free',
               'google/gemini-2.0-flash-exp:free',
               'qwen/qwen3-32b-04-28:free',
               'mistralai/mistral-nemo:free',
               'qwen/qwen3-14b-04-28:free',
               'qwen/qwen3-235b-a22b-04-28:free']

cheap_models = ['google/gemini-2.0-flash-001',
                'google/gemini-2.5-flash-preview-05-20',
                'google/gemini-2.5-flash-lite-preview-06-17',
                'google/gemini-2.5-flash-preview-04-17',
                'openai/gpt-4o-mini']

expensive_models = ['deepseek/deepseek-r1-0528',
                    'anthropic/claude-3-5-haiku',
                    'nousresearch/hermes-3-llama-3.1-405b',
                    'sao10k/l3.1-euryale-70b',
                    'meta-llama/llama-3.1-405b-instruct']

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

def generate_prompt(game_state: list, action_state: int, comparator_state: int, score_state: int):
    """
    Generates prompts for models which describe the rules of absurd soccer, provide commentary for a game, and asks for the outcome of that game
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    """
    prompt = ""
    prompt = f"Absurd soccer is played by two teams of {game_symbols[game_state[0]]}s. In one match of this game, each team of {game_symbols[game_state[0]]} takes a turn to shoot {game_symbols[game_state[1]]} five times at a {game_symbols[game_state[2]]}. A {game_symbols[game_state[0]]} can shoot only once in a match. When one team shoots, the other team defends the {game_symbols[game_state[2]]}. If the {game_symbols[game_state[0]]} that makes the shot {action_symbols[action_state]} the {game_symbols[game_state[2]]}, their team gets one {score_symbols[score_state]}. At the end of the match, the team having the {comparator_symbols[comparator_state]} {score_symbols[score_state]}s wins.\n\n"
    prompt += "Here is the match commentary for a game of absurd soccer:\n\n"
    game, answer = generate_game(game_state, action_state, comparator_state)
    prompt += game
    prompt += "\nWho won the game? Answer 'team A' if team A wins, 'team B' if team B wins, and 'both' if both teams wins. Please place your answer within two curly brackets (ex. {team A})."
    return prompt, answer

def run_sim(api_key: str, num_sims: int, game_state: list, action_state: int, comparator_state: int, score_state: int, price_of_models: str):
    """
    Tests each model's ability to evaluate a game of absurd soccer using the generate_prompt function
    - api_key: OpenRouter API Key
    - game_state: determines how the game symbols (["player", "ball", "net"]) are arranged
    - action_state: determines how the action symbols (["hits", "misses"]) are arranged
    - comparator_state: determines how the comparator symbols (["most", "least"]) are arranged
    - score_state: determines how the score symbols (["score", "point", "car", "ice-cream"]) are arranged
    - price_of_models: determines what models will be tested on.
        - "free": free models
        - "cheap": $0.1-$0.2 per token
        - "expensive": $0.5-$1 per token
    """
    if price_of_models == "cheap":
        models = cheap_models
    elif price_of_models == "expensive":
        models = expensive_models
    else:
        models = free_models

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
        prompt, answer = generate_prompt(game_state, action_state, comparator_state, score_state)
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
            data[model].append(re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()))
            if data['answer'][-1].lower() in data[model][-1].lower():
                total_results[model] += 1
        
        print("Generated game", str(i))

    data['game #'].append('total')
    data['prompt'].append(None)
    data['answer'].append(None)
    for model in models:
        data[model].append(total_results[model] / num_sims)

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
