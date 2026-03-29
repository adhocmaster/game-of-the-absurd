import random
from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
import re
import statistics
import math

prize_symbols = ["goat", "cow"]
order_symbols = ["first", "last"]

default_model_list = ['anthropic/claude-opus-4.6',
               'anthropic/claude-sonnet-4.6',
               'x-ai/grok-4.1-fast',
               'google/gemini-3-flash-preview',
               'openai/gpt-5.4',
               'deepseek/deepseek-v3.2',
               'qwen/qwen3-max',
               'minimax/minimax-m2.7',
               'google/gemini-2.5-flash-lite',
               'google/gemini-2.5-flash',
               'xiaomi/mimo-v2-omni',
               'moonshotai/kimi-k2.5',
               'openai/gpt-oss-120b',
               'google/gemini-3.1-flash-lite-preview',
               'openai/gpt-4o-mini',
               'xiaomi/mimo-v2-flash',
               'openai/gpt-5-mini',
               'x-ai/grok-4-fast',
               'qwen/qwen3-235b-a22b-2507',
               'openai/gpt-4.1-mini',
               'openai/gpt-5.4-nano',
               'deepseek/deepseek-chat-v3-0324',
               'mistralai/mistral-nemo',
               'openai/gpt-5-nano',
               'openai/gpt-oss-20b']

def save_results_to_file(df, identifier):
    try:
        os.remove("./"+identifier + '.csv')
    except:
        pass
    df.to_csv(identifier + '.csv', index=False)

def turn_ruleset_to_settings(ruleset: str):
    prize_symbol = "goat"
    order_symbol = "first"

    if ruleset not in ["REAL", "COW", "LAST"]:
        quit()
    
    if ruleset == "COW":
        prize_symbol = "cow"
    if ruleset == "LAST":
        order_symbol = "last"
    
    return prize_symbol, order_symbol

def generate_events_task(api_key: str, num_sims: int, ruleset: str, model_names, file_name: str):
    prize_symbol, order_symbol = turn_ruleset_to_settings(ruleset)

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
            data[model + '_values'] = []
            data[model + '_outcome'] = []
    
        df = pd.DataFrame(data)
    
    original_length = len(df)
    if original_length > num_sims:
        return
    
    for i in range(num_sims-original_length):
        new_row = {}
        if order_symbol == "last":
            outcome = random.randint(3,5)
        else:
            outcome = random.randint(1,3)


        prompt = f"Five players, numbered 1 through 5, play a game. In front of them are five unopened doors, numbered 1 through 5. Doors 1, 3, and 5 have a {prize_symbol} behind it. Doors 2 and 4 do not. Player 1 chooses an unopened door and opens it. Then, player 2 chooses a door that has not been opened and opens it. Players 3, 4, and 5 do the same, in that order. Once all five players have finished opening the doors, the player who wins the game is the one who opened a door with a goat behind it first. \n\n Your task is to come up with a list of numbers a_1, a_2, a_3, a_4, a_5, such that if player 1 opens door a_1, player 2 opens door a_2, . . . , and player 5 opens door a_5, then player {outcome} wins. Note that a_1, a_2, a_3, a_4, a_5 must all be different from each other and that a_n must be a number between 1 and 5. Please work out your reasoning process for the answer, and format the list within brackets (ex. {{3,1,2,5,4}})"
        
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

            list_of_nums = re.sub(r'[^a-zA-Z0-9,]+', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip().lower()).split(",")

            all_5 = False
            if (x in list_of_nums for x in [1, 2, 3, 4, 5]):
                all_5 = True

            while completion.choices == None or len(list_of_nums) != 5 or not all_5:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                list_of_nums = re.sub(r'[^a-zA-Z0-9,]+', '', completion.choices[0].message.content.split("{")[-1].split("}")[0].strip().lower()).split(",")
            
            winner = -1
            if order_symbol == "last":
                for i in range(5):
                    if list_of_nums[i] in [1, 3, 5]:
                        winner = int(list_of_nums[4-i].strip())
            else:
                for i in range(5):
                    if list_of_nums[i] in [1, 3, 5]:
                        winner = int(list_of_nums[i].strip())

            new_row[model + '_response'] = [completion.choices[0].message.content]
            new_row[model + '_values'] = [list_of_nums]
            new_row[model + '_outcome'] = [winner]
        
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

        new_row[model + '_values'] = [None]
        new_row[model + '_response'] = [None]
        new_row[model + '_outcome'] = [total_results[model] / num_sims]

    new_row = pd.DataFrame(new_row)
    df = pd.concat([df, new_row])
    save_results_to_file(df, file_name)

def run_all_models(task: str, api_key: str, num_sims: int, ruleset: str, model_names):
    if task == "GE":
        for i in range(num_sims):
            generate_events_task(api_key, num_sims, ruleset, model_names, "ge_"+ruleset)

def run_all_rulesets(task: str, api_key: str, num_sims: int, model_names):
    all_rulesets = ["REAL", "COW", "LAST"]

    if not os.path.exists(task):
        os.mkdir(task)

    os.chdir(task)

    for r in all_rulesets:
        run_all_models(task, api_key, num_sims, r, model_names)
    
    os.chdir('..')

def run_full_exp(folder_name, api_key, num_sims, model_names):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)
    all_tasks = ["GE"]
    for t in all_tasks:
        run_all_rulesets(t, api_key, num_sims, model_names)