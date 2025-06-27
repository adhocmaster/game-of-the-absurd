import random
from openai import OpenAI
import os

# IN ORDER TO RUN THIS PROGRAM, YOU WILL NEED AN OPEN ROUTER API KEY

# INITIAL PARAMETERS

game_symbols = ["player", "ball", "net"]
action_symbols = ["hits", "misses"]
comparator_symbols = ["most", "least"]
score_symbols = ["score", "point", "car", "ice-cream"]

model_names = ["google/gemini-2.5-flash", "anthropic/claude-3-7-sonnet-20250219", "openai/chatgpt-4o-latest", "meta-llama/llama-4-scout-17b-16e-instruct:free", "x-ai/grok-3-mini"]
game_settings = [0, 1, 2]
action_settings = 0
comparator_settings = 1
score_settings = 1

# GAME GENERATION

def generate_game(game_state: list, action_state: int, comparator_state: int):
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
    prompt = ""
    prompt = f"Absurd soccer is played by two teams of {game_symbols[game_state[0]]}s. In one match of this game, each team of {game_symbols[game_state[0]]} takes a turn to shoot {game_symbols[game_state[1]]} five times at a {game_symbols[game_state[2]]}. A {game_symbols[game_state[0]]} can shoot only once in a match. When one team shoots, the other team defends the {game_symbols[game_state[2]]}. If the {game_symbols[game_state[0]]} that makes the shot {action_symbols[action_state]} the {game_symbols[game_state[2]]}, their team gets one {score_symbols[score_state]}. At the end of the match, the team having the {comparator_symbols[comparator_state]} {score_symbols[score_state]}s wins.\n\n"
    prompt += "Here is the match commentary for a game of absurd soccer:\n\n"
    game, answer = generate_game(game_state, action_state, comparator_state)
    prompt += game
    prompt += "\nWho won the game? Answer 'team A' if team A wins, 'team B' if team B wins, and 'both' if both teams wins. Please place your answer within two curly brackets (ex. {team A})."
    return prompt, answer


def run_sim(models: list, num_sims: int, game_state: list, action_state: int, comparator_state: int, score_state: int):
    results = []
    for model in models:
        results.append(0)
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        for i in range(num_sims):
            prompt, answer = generate_prompt(game_state, action_state, comparator_state, score_state)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            if completion.choices != None:
                if completion.choices[0].message.content.split("{")[-1].split("}")[0].strip() == answer:
                    results[-1] += 1

        results[-1] /= num_sims

    return results

results = run_sim(model_names, 20, game_settings, action_settings, comparator_settings, score_settings)

for name, result in zip(model_names, results):
    print(name + ": "  + result)
