

# Can Large Language Model Agents Simulate Human Trust Behaviors?

- **Paper** : [Read our paper](https://arxiv.org/abs/2402.04559)
- **Project Website**: [Visit the project website](https://www.camel-ai.org/research/agent-trust)
- **Online Demo**: [Trust Game Demo](https://huggingface.co/spaces/yitianlian/trust_game_demo) & [Repeated Trust Game Demo](https://huggingface.co/spaces/yitianlian/repeated_trust_game_demo)

Our research investigates the simulation of human trust behaviors through the use of large language model agents. We leverage the foundational work of the Camel Project, acknowledging its significant contributions to our research. For further information about the Camel Project, please visit [Camel AI](https://github.com/camel-ai/camel).

## Setting Up the Experiment Environment

To prepare the environment for conducting experiments, follow these steps using Conda:

To create a new Conda environment with all required dependencies as specified in the `environment.yaml` file, use:

```bash
conda env create -f environment.yaml
```

Alternatively, you can set up the environment manually as follows:

```bash
conda create -n agent-trust python=3.10
pip install -r requirements.txt
```

### Running Trust Games Demos Locally

This guide provides instructions on how to run the trust games demos on your local machine. We offer two types of trust games: non-repeated and repeated. Follow the steps below to execute each demo accordingly.

#### Non-Repeated Trust Game Demo

To run the non-repeated trust game demo, use the following command in your terminal:

```bash
python agent_trust/no_repeated_demo.py
```

#### Repeated Trust Game Demo

For the repeated trust game demo, execute this command:

```bash
python agent_trust/repeated_demo.py
```

Running this command will start the demo where the trust game is played repeatedly, illustrating how trust can evolve over repeated interactions.

Ensure you have the required environment set up and dependencies installed before running these commands. Enjoy exploring the trust dynamics in both scenarios!
## Experiment Code Overview

The experiment code is primarily located in `agent_trust/all_game_person.py`, which contains the necessary implementations for executing the trust behavior experiments with large language models.

### Open-Source Models

We utilize the [FastChat](https://github.com/lm-sys/FastChat) Framework for smooth interactions with open-source models. For comprehensive documentation, refer to the [FastChat GitHub repository](https://github.com/lm-sys/FastChat).

### Game Prompts

Game prompts are vital for our experiments and are stored in `agent_trust/prompt`. These JSON files provide the prompts used throughout the experiments, ensuring transparency and reproducibility.

## Running the Experiments

### No Repeated Trust Game

For scenarios where the trust game is not repeated, execute the experiment by running the `run_exp` function in the `all_game_person.py` file. Ensure you adjust the `model_list` and other parameters according to your experiment's specifics.

### Repeated Trust Game Experiment

For experiments involving repeated trust games, use the `multi_round_exp` function in the `all_game_person.py` file. This function is specifically designed for use with GPT-3.5-16k and GPT-4 models.

### Web Interface for Experiments

To access a web interface for running the experiments (demo), execute `agent_trust/no_repeated_demo.py` or `agent_trust/repeated_demo.py`. This provides a user-friendly interface to interact with the experiment setup. You can also visit our online demo websites: [Trust Game Demo](https://huggingface.co/spaces/yitianlian/trust_game_demo) & [Repeated Trust Game Demo](https://huggingface.co/spaces/yitianlian/repeated_trust_game_demo)

