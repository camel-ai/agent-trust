


# Can Large Language Model Agents Simulate Human Trust Behaviors?

Our research explores the simulation of human trust behaviors using large language model agents. The foundation of our code is the Camel Project, which has significantly contributed to our work's success. We extend our gratitude to the creators of this remarkable project. For more information on **Camel**, visit [Camel AI](https://github.com/camel-ai/camel).

## Setting Up the Experiment Environment

To begin, you will need to set up the appropriate environment for running the experiments. This can be done easily using Conda:

```bash
conda env create -f environment.yaml
```

This command creates a new Conda environment with all the necessary dependencies installed, as specified in the `environment.yaml` file.

## Experiment Code

The core of our experiment can be found in `examples/agent_trust/all_game_person.py`. This file contains the implementation necessary to conduct the trust behavior experiments with large language models.

### Open-Source Models

Our experiments leverage the [FastChat](https://github.com/lm-sys/FastChat) Framework for seamless interaction with open-source models. Detailed documentation for this framework is available at the [FastChat GitHub repository](https://github.com/lm-sys/FastChat).

### Game Prompts

The prompts used in our game settings are located in `examples/agent_trust/prompt`. These JSON files detail the exact prompts employed throughout the experiments, facilitating a transparent and reproducible research process.

## Running the Experiments

### No Repeated Trust Game

To execute the experiment for scenarios where the trust game is not repeated, run the `run_exp` function in the `all_game_person.py` file. Before running, ensure to adjust the `model_list` and other parameters to match your specific game setting.

### Repeated Trust Game Experiment

For experiments involving repeated trust games, utilize the `multi_round_exp` function within the `all_game_person.py` file. It's important to note that this function is designed exclusively for use with GPT-3.5-16k and GPT-4 models.

