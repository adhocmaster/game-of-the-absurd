import absurd_soccer

# Everything is normal, except the team having the least points wins

symbol_settings = [0, 1, 2]
action_choice = 0
comparator_choice = 1
score_settings = 1
price = 'expensive'

results = absurd_soccer.run_sim(os.environ.get("OPENROUTER_API_KEY"), 100, symbol_settings, action_choice, comparator_choice, score_settings, price)

absurd_soccer.save_results_to_file(results, price + "_least_points_wins")
