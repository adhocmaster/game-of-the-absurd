import random
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
import re
import statistics
import math

default_model_list = ['anthropic/claude-opus-4.6',
               'anthropic/claude-sonnet-4.6',
               #'x-ai/grok-4.1-fast',
               'google/gemini-3-flash-preview',
               'openai/gpt-5.4',
               'deepseek/deepseek-v3.2',
               'qwen/qwen3-max',
               'minimax/minimax-m2.7',
               'google/gemini-2.5-flash-lite',
               'google/gemini-2.5-flash',
               'xiaomi/mimo-v2-omni',
               'moonshotai/kimi-k2.5',
               'google/gemini-3.1-flash-lite-preview',
               'openai/gpt-4o-mini',
               'xiaomi/mimo-v2-flash',
               'openai/gpt-5-mini',
               #'x-ai/grok-4-fast',
               'qwen/qwen3-235b-a22b-2507',
               'openai/gpt-4.1-mini',
               'openai/gpt-5.4-nano',
               'deepseek/deepseek-chat-v3-0324',
               'mistralai/mistral-nemo',
               'openai/gpt-5-nano']

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

def do_random(api_key: str, num_sims: int, model_names, file_name: str):
    if model_names == "default":
        models = default_model_list
    elif type(model_names) is list:
        models = model_names
    else:
        # TODO: CHANGE STRINGS
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return

    for i in range(num_sims-original_length):
        random_players = random.sample(['A','C','D'],3)
        random_doors = random.sample([1,2,3,4,5],3)
        
        outcome = 'none'
        for j in range(3):
            if random_doors[j] in [1, 3]:
                outcome = random_players[j]
                break


        new_row = {}

        prompt = f"Three players (player A, player C, player D) play the following game: in front of them are five unopened doors, numbered 1 through 5. Doors 1 and 3 have a goat behind it. Doors 2, 4, and 5 do not. The players then open the doors in some order, and the player which opens a door with a goat behind it first wins the game. If no player opens a door with a goat behind it, no one wins.\n\nSuppose the three players open some of the doors in the following order:\n\nPlayer {random_players[0]} opens door {random_doors[0]}\nPlayer {random_players[1]} opens door {random_doors[1]}\nPlayer {random_players[2]} opens door {random_doors[2]}\n\nYour task is to determine who the winner of this game is. Answer 'A' if player A wins, 'C' if player C wins, 'D' if player D wins, and 'none' if no players win. Please work out your reasoning process for the answer, and place your answer within two curly brackets (ex. {{A}})." 

        new_row['game #'] = [i+original_length]
        new_row['prompt'] = [prompt]
        new_row['answer'] = [outcome]

        print("Generating game", str(i+original_length))  

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
            while completion.choices == None or (completion.choices[0].message.content != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()).lower() not in ['a','c','d','none']):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            
            new_row[model + '_response'] = [completion.choices[0].message.content]
            new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip())]

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
        for i in range(num_sims):
            if type(list(df[model + '_outcome'])[i]) == str:
                if list(df['answer'])[i].lower() == list(df[model + '_outcome'])[i].lower():
                    total_results[model] += 1

        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)

def do_random_last(api_key: str, num_sims: int, model_names, file_name: str):
    if model_names == "default":
        models = default_model_list
    elif type(model_names) is list:
        models = model_names
    else:
        # TODO: CHANGE STRINGS
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return

    for i in range(num_sims-original_length):
        random_players = random.sample(['A','C','D'],3)
        random_doors = random.sample([1,2,3,4,5],3)
        
        outcome = 'none'
        for j in range(3):
            if random_doors[2-j] in [1, 3]:
                outcome = random_players[2-j]
                break


        new_row = {}

        prompt = f"Three players (player A, player C, player D) play the following game: in front of them are five unopened doors, numbered 1 through 5. Doors 1 and 3 have a goat behind it. Doors 2, 4, and 5 do not. The players then open the doors in some order, and the player which opens a door with a goat behind it last wins the game. If no player opens a door with a goat behind it, no one wins.\n\nSuppose the three players open some of the doors in the following order:\n\nPlayer {random_players[0]} opens door {random_doors[0]}\nPlayer {random_players[1]} opens door {random_doors[1]}\nPlayer {random_players[2]} opens door {random_doors[2]}\n\nYour task is to determine who the winner of this game is. Answer 'A' if player A wins, 'C' if player C wins, 'D' if player D wins, and 'none' if no players win. Please work out your reasoning process for the answer, and place your answer within two curly brackets (ex. {{A}})." 

        new_row['game #'] = [i+original_length]
        new_row['prompt'] = [prompt]
        new_row['answer'] = [outcome]

        print("Generating game", str(i+original_length))  

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
            while completion.choices == None or (completion.choices[0].message.content != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()).lower() not in ['a','c','d','none']):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            
            new_row[model + '_response'] = [completion.choices[0].message.content]
            new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip())]

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
        for i in range(num_sims):
            if type(list(df[model + '_outcome'])[i]) == str:
                if list(df['answer'])[i].lower() == list(df[model + '_outcome'])[i].lower():
                    total_results[model] += 1

        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)


def do_order(api_key: str, num_sims: int, ruleset: str, model_names, file_name: str):
    pick_events = [[("D", 3), ("A", 4), ("C", 5)],
              [("C", 5), ("A", 1), ("D", 2)],
              [("D", 5), ("A", 3), ("C", 4)],
              [("D", 1), ("C", 3), ("A", 4)],
              [("C", 5), ("D", 4), ("A", 1)],
              [("A", 4), ("D", 3), ("C", 1)],
              [("C", 5), ("A", 1), ("D", 4)],
              [("D", 3), ("C", 1), ("A", 4)],
              [("D", 1), ("A", 2), ("C", 3)],
              [("D", 2), ("A", 5), ("C", 1)],
              [("C", 2), ("A", 3), ("D", 4)],
              [("A", 2), ("C", 5), ("D", 1)],
              [("A", 1), ("C", 5), ("D", 4)],
              [("C", 5), ("A", 1), ("D", 2)],
              [("C", 1), ("A", 2), ("D", 4)],
              [("C", 2), ("D", 4), ("A", 3)],
              [("C", 1), ("D", 2), ("A", 3)],
              [("C", 4), ("A", 3), ("D", 1)]]
    
    if model_names == "default":
        models = default_model_list
    elif type(model_names) is list:
        models = model_names
    else:
        # TODO: CHANGE STRINGS
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return

    for i in range(num_sims-original_length):
        pick_event = random.choice(pick_events)
        random.shuffle(pick_event)
        
        outcome = 'none'
        for j in range(3):
            if pick_event[j][1] in [1, 3]:
                outcome = pick_event[j][0]
                break


        new_row = {}

        prompt = f"Three players (player A, player C, player D) play the following game: in front of them are five unopened doors, numbered 1 through 5. Doors 1 and 3 have a goat behind it. Doors 2, 4, and 5 do not. The players then open the doors in some order, and the player which opens a door with a goat behind it first wins the game. If no player opens a door with a goat behind it, no one wins.\n\nSuppose the three players open some of the doors in the following order:\n\nPlayer {pick_event[0][0]} opens door {pick_event[0][1]}\nPlayer {pick_event[1][0]} opens door {pick_event[1][1]}\nPlayer {pick_event[2][0]} opens door {pick_event[2][1]}\n\nYour task is to determine who the winner of this game is. Answer 'A' if player A wins, 'C' if player C wins, 'D' if player D wins, and 'none' if no players win. Please work out your reasoning process for the answer, and place your answer within two curly brackets (ex. {{A}})." 

        new_row['game #'] = [i+original_length]
        new_row['prompt'] = [prompt]
        new_row['answer'] = [outcome]

        print("Generating game", str(i+original_length))  

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
            while completion.choices == None or (completion.choices[0].message.content != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()).lower() not in ['a','c','d','none']):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            
            new_row[model + '_response'] = [completion.choices[0].message.content]
            new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip())]

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
        for i in range(num_sims):
            if type(list(df[model + '_outcome'])[i]) == str:
                if list(df['answer'])[i].lower() == list(df[model + '_outcome'])[i].lower():
                    total_results[model] += 1

        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)

def do_order_no_win(api_key: str, num_sims: int, ruleset: str, model_names, file_name: str):
    pick_events = [[("D", 2), ("A", 5), ("C", 4)]]
    
    if model_names == "default":
        models = default_model_list
    elif type(model_names) is list:
        models = model_names
    else:
        # TODO: CHANGE STRINGS
        print("model_names variable must either be a specific string ('free', 'cheap', 'expensive', 'reasoning') or a list of model names")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

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
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return

    for i in range(num_sims-original_length):
        pick_event = random.choice(pick_events)
        random.shuffle(pick_event)
        
        outcome = 'none'
        for j in range(3):
            if pick_event[j][1] in [1, 3]:
                outcome = pick_event[j][0]
                break


        new_row = {}

        prompt = f"Three players (player A, player C, player D) play the following game: in front of them are five unopened doors, numbered 1 through 5. Doors 1 and 3 have a goat behind it. Doors 2, 4, and 5 do not. The players then open the doors in some order, and the player which opens a door with a goat behind it first wins the game. If no player opens a door with a goat behind it, no one wins.\n\nSuppose the three players open some of the doors in the following order:\n\nPlayer {pick_event[0][0]} opens door {pick_event[0][1]}\nPlayer {pick_event[1][0]} opens door {pick_event[1][1]}\nPlayer {pick_event[2][0]} opens door {pick_event[2][1]}\n\nYour task is to determine who the winner of this game is. Answer 'A' if player A wins, 'C' if player C wins, 'D' if player D wins, and 'none' if no players win. Please work out your reasoning process for the answer, and place your answer within two curly brackets (ex. {{A}})." 

        new_row['game #'] = [i+original_length]
        new_row['prompt'] = [prompt]
        new_row['answer'] = [outcome]

        print("Generating game", str(i+original_length))  

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
            while completion.choices == None or (completion.choices[0].message.content != None and re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip()).lower() not in ['a','c','d','none']):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            
            new_row[model + '_response'] = [completion.choices[0].message.content]
            new_row[model + '_outcome'] = [re.sub(r'[^a-zA-Z0-9 ]', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip())]

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
        for i in range(num_sims):
            if type(list(df[model + '_outcome'])[i]) == str:
                if list(df['answer'])[i].lower() == list(df[model + '_outcome'])[i].lower():
                    total_results[model] += 1

        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)

def run_all_models(task: str, api_key: str, num_sims: int, ruleset: str, model_names):
    if task == "DO_RANDOM":
        for i in range(num_sims):
            do_random(api_key, num_sims, model_names, "do_random")
    elif task == "DO_ORDER":
        for i in range(num_sims):
            do_order(api_key, num_sims, ruleset, model_names, "do_order")
    elif task == "DO_ORDER_NO_WIN":
        for i in range(num_sims):
            do_order_no_win(api_key, num_sims, ruleset, model_names, "do_order")
    elif task == "DO_RANDOM_LAST":
        for i in range(num_sims):
            do_order_no_win(api_key, num_sims, ruleset, model_names, "do_random_last")

def run_all_rulesets(task: str, api_key: str, num_sims: int, model_names):
    if not os.path.exists(task):
        os.mkdir(task)

    os.chdir(task)

    run_all_models(task, api_key, num_sims, "", model_names)
    
    os.chdir('..')

def run_full_exp(folder_name, api_key, num_sims, model_names):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)

    all_tasks = ["DO_RANDOM", "DO_ORDER", "DO_ORDER_NO_WIN", "DO_RANDOM_LAST"]
    for t in all_tasks:
        run_all_rulesets(t, api_key, num_sims, model_names)
