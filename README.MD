# Absurd World: A Simple Yet Powerful Method to Absurdify the Real-world for Probing LLM Reasoning Capabilities

This is the codebase for the paper "Absurd World: A Simple Yet Powerful Method to Absurdify the Real-world for Probing LLM Reasoning Capabilities," submitted for review to ICML 2026.

# Requirements

To run this codebase, you will need an OpenRouter API key. 

# Setup

In a directory, run the following command:
```
git clone https://github.com/adhocmaster/game-of-the-absurd.git
```

Then, within the same directory, in a python file or notebook, replace OPENROUTER_KEY with your OpenRouter API key and run the following lines:
```
import absurd_soccer
absurd_soccer.run_full_exp("Project_name", OPENROUTER_KEY, 100, "all")
```

This will run the same experiment that we performed in our paper.