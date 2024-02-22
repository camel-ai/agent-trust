import copy
import json
import os
import sys

import gradio as gr
import openai
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig, OpenSourceConfig
from camel.messages import BaseMessage
from camel.types import ModelType, RoleType
from exp_model_class import ExtendedModelType

open_model_path_dict = {
    ModelType.VICUNA: "lmsys/vicuna-7b-v1.3",
    ModelType.LLAMA_2: "meta-llama/Llama-2-7b-chat-hf",
}
front = "you are a person not an ai model."


def str_mes(content):
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )


def gpt3_res(prompt, model_name="text-davinci-003", temperature=1):
    response = openai.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=1500,
    )
    return response.choices[0].text.strip()


def get_res_for_visible(
    role,
    first_message,
    game_type,
    api_key,
    model_type=ExtendedModelType.GPT_4,
    extra_prompt="",
    temperature=1.0,
    player_demographic=None,
):
    content = ""
    if api_key is not None or api_key != "":
        openai.api_key = api_key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    extra_prompt += "Your answer needs to include the content about your BELIEF, DESIRE and INTENTION."
    if "game" in game_type.lower():
        extra_prompt += "You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."
    else:
        extra_prompt += "You must end with 'Finally, I will choose ___' ('Trust' or 'not Trust' are required in the spaces)."
    extra_prompt += front

    role = str_mes(role + extra_prompt)
    if player_demographic is not None:
        first_message = first_message.replace(
            "player", player_demographic+" player")
    first_message = str_mes(first_message)
    if model_type in [
        ExtendedModelType.INSTRUCT_GPT,
        ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
    ]:
        message = role.content + first_message.content + extra_prompt
        final_res = str_mes(gpt3_res(message, model_type.value, temperature))
    else:
        role = str_mes(role.content + extra_prompt)
        model_config = ChatGPTConfig(temperature=temperature)
        if model_type in [
            ModelType.VICUNA,
            ModelType.LLAMA_2,
        ]:
            open_source_config = dict(
                model_type=model_type,
                model_config=OpenSourceConfig(
                    model_path=open_model_path_dict[model_type],
                    server_url="http://localhost:8000/v1",
                    api_params=ChatGPTConfig(temperature=temperature),
                ),
            )
            agent = ChatAgent(
                role, output_language="English", **(open_source_config or {})
            )
        else:
            agent = ChatAgent(
                role,
                model_type=model_type,
                output_language="English",
                model_config=model_config,
            )
        final_all_res = agent.step(first_message)
        final_res = final_all_res.msg
    content += final_res.content

    return content


sys.path.append("../..")

file_path_character_info = 'prompt/character_2.json'
file_path_game_prompts = 'prompt/person_all_game_prompt.json'

with open(file_path_character_info, 'r') as file:
    character_info = json.load(file)

# Load game prompts
with open(file_path_game_prompts, 'r') as file:
    game_prompts = json.load(file)

# Extract character names and information
characters = [f'Trustor Persona {i}' for i in range(
    1, len(character_info) + 1)]
character_info = {f'Trustor Persona {i}': info for i, info in enumerate(
    character_info.values(), start=1)}

# Extract game names and prompts
game_prompts = {
    prompt[0]: prompt[-1] for i, prompt in enumerate(game_prompts.values(), start=1)}
games = list(game_prompts.keys())
print(games)

model_dict = {
    'gpt-3.5-turbo-0613': ExtendedModelType.GPT_3_5_TURBO_0613,
    'gpt-3.5-turbo-16k-0613': ExtendedModelType.GPT_3_5_TURBO_16K_0613,
    'gpt-4': ExtendedModelType.GPT_4,
    'text-davinci-003': ExtendedModelType.INSTRUCT_GPT,
    'gpt-3.5-turbo-instruct': ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
    # 'vicuna': ModelType.VICUNA,
    # 'llama-2': ModelType.LLAMA_2,
}
game_tree_images = {
    "Dictator_Game": "game_tree/dictator_game_game_tree.png",
    "Trust_Game": "game_tree/Trust_game_game_tree.png",
    "map_risky_dictator_problem": "game_tree/risky_dictator_game_game_tree.png",
    "map_trust_problem": "game_tree/map_trust_game_game_tree.png",
    "lottery_problem_people": "game_tree/lottery_people_game_tree.png",
    "lottery_problem_gamble": "game_tree/lottery_gamble_game_tree.png"
}

models = list(model_dict.keys())


def update_char_info(char):
    return character_info.get(char, "No information available.")


def update_game_prompt(game):
    return game_prompts.get(game, "No prompt available.")


def process_submission(character, game, api_key=None,  model="gpt-3.5-turbo-0613",  extra_prompt="", temperature=1.0, player_demographic=None,):
    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
    return get_res_for_visible(character_info.get(character, ""), game_prompts.get(game, "No prompt available."), game, api_key, model_dict[model], extra_prompt, temperature, player_demographic)


def update_game_image(game_name):

    image_path = game_tree_images.get(game_name, None)

    return image_path


with gr.Blocks() as app:
    game_introduction = gr.Textbox(
        label="Instruction", value="""1. You should select the persona for the trustor and the type of game.\n
2. You need to fill in your OpenAI API Key.\n
3. If you fill in 'Extra Prompt for Trustor', this prompt will be the additional system prompt to the trustor.\n
4. You can fill in the trustee player's demographics, such as race or gender.\n
5. If you want reset the conversation, please refresh this page.""")
    with gr.Row():
        char_dropdown = gr.Dropdown(
            choices=characters, label="Select Trustor Persona", value=characters[0])
        game_dropdown = gr.Dropdown(
            choices=games, label="Select Game")
    char_info_display = gr.Textbox(
        label="Trustor Persona Info", value=character_info[characters[0]])
    with gr.Row():
        game_prompt_display = gr.Textbox(
            label="Game Prompt", value=game_prompts["Trust_Game"])
        game_image_display = gr.Image(
            label="Game Image")

    api_key_input = gr.Textbox(
        label="OpenAI API Key", placeholder="Enter your OpenAI API Key here")
    model_dropdown = gr.Dropdown(
        choices=models, label="Select Model", value=models[0])
    extra_prompt_input = gr.Textbox(
        label="Extra Prompt for Trustor", placeholder="Enter any additional prompt here (Optional)")
    temperature_slider = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.01, label="Temperature", value=1.0)
    player_demographic_input = gr.Textbox(
        label="Trustee Player Demographic", placeholder="Enter trustee player demographic info here (Optional)")
    submit_button = gr.Button("Submit")
    result_display = gr.Textbox(label="Result")

    # 更新显示信息
    char_dropdown.change(
        update_char_info, inputs=char_dropdown, outputs=char_info_display)
    game_dropdown.change(update_game_prompt,
                         inputs=game_dropdown, outputs=game_prompt_display)
    game_dropdown.change(
        update_game_image, inputs=game_dropdown, outputs=game_image_display)

    submit_button.click(
        process_submission,
        inputs=[char_dropdown, game_dropdown, api_key_input, model_dropdown,
                extra_prompt_input, temperature_slider, player_demographic_input],
        outputs=result_display
    )

app.launch()
