import json
import os
import sys

import gradio as gr
import openai
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.types.enums import RoleType

from exp_model_class import ExtendedModelType
from multi_round_person import (classmate, extract_unique_decimal, match_and_compare_numbers_v2,
                                str_mes)

openai.api_key = os.getenv("OPENAI_API_KEY")
roles = ["trustor", "trustee", "None"]
model_dict = {
    'gpt-3.5-turbo-16k-0613': ExtendedModelType.GPT_3_5_TURBO_16K_0613,
    'gpt-4': ExtendedModelType.GPT_4,
}
sys.path.append("../..")
with open(
    r"prompt/trust_game_round_prompt.json",
    "r",
) as f:
    prompt = json.load(f)
file_path_character_info = 'prompt/character_2.json'
models = list(model_dict.keys())
with open(file_path_character_info, 'r') as file:
    character_info = json.load(file)

user_input_prompt_template = "I will give ${k}"

# Extract character names and information
characters = [f'Persona {i}' for i in range(1, len(character_info) + 1)]
characters.insert(0, "Human(You)")
character_info = {f'Persona {i}': info for i, info in enumerate(
    character_info.values(), start=1)}
character_info["Human(You)"] = "You"

initial_dialog_history = []
initial_round_num = 0

like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""

initial_chat_agent = None


def update_char_info(char):
    return character_info.get(char, "No information available.")


def classmate_with_human(
    player_1,
    player_2,
    user_input,
    user_role,
    first_round,
    first_prompt,
    second_prompt,
    k,
):
    first_round_prompt = "This is the first round, answer the question."
    money_prompt = "Now,the another player give you {give} dollars,and You receive {N} dollars,the player left {left} dollars now. How much will you give back to the another player"
    return_money_prompt = "In last round ,You give the another player {give} dollars, The another player receive {receive} dollars, and The another player return you {N} dollars.Last round you left {left} dollars.This round is begin. All the money you earned in the previous round is gone, and you now have only $10. How much will you give to the another player?"
    player_2_end_prompt = "In last round, the another player give you {give} dollars, you receive {receive} dollars, and you return the another player {N} dollars.Last round you left {left} dollars. This round is begin. All the money you earned in the previous round is gone."
    grantee = "Your answer needs to include the content and analysis about your BELIEF, DESIRE and INTENTION. You should include your thought. You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."
    res = []
    cri_agent = ChatAgent(
        BaseMessage(
            role_name="critic",
            role_type=RoleType.ASSISTANT,
            meta_dict={},
            content='How much would this person pay the other student? Only response with a specific price number like "5"!Don\'t response with a sentence',
        ),
        output_language="English",
        # model=ModelType.STUB,
    )

    if first_round:
        if user_role == "trustor":
            given_num = user_input
            player_1_response = str_mes(
                user_input_prompt_template.format(k=given_num))
        else:
            player_1_response = player_1.step(
                str_mes(first_round_prompt + grantee)).msgs[0]
            ans = match_and_compare_numbers_v2(player_1_response.content)
            if ans:
                given_num = ans
            else:
                given_num = extract_unique_decimal(
                    cri_agent.step(
                        str_mes(player_1_response.content)).msgs[0].content
                )

        money_prompt = money_prompt.format(
            give=given_num, N=given_num * k, left=10 - given_num
        )
        if user_role == "trustee":
            player_2_response = str_mes(
                user_input_prompt_template.format(k=given_num * k))
        else:
            player_2_response = player_2.step(
                str_mes(money_prompt + grantee)).msgs[0]
    else:
        if user_role == "trustor":
            given_num = user_input
            player_1_response = str_mes(
                user_input_prompt_template.format(k=given_num))
        else:
            player_1_response = player_1.step(
                str_mes(first_prompt + grantee)).msgs[0]
            # print("player 1 input", first_prompt)
            # print("Player_1_res", player_1_response.content)
            ans = match_and_compare_numbers_v2(player_1_response.content)
            if ans:
                given_num = ans
            else:
                given_num = extract_unique_decimal(
                    cri_agent.step(
                        str_mes(player_1_response.content)).msgs[0].content
                )
        money_prompt = money_prompt.format(
            give=given_num, N=given_num * k, left=10 - given_num
        )
        if user_role == "trustee":
            player_2_response = str_mes(
                user_input_prompt_template.format(k=given_num * k))
        else:
            player_2_response = player_2.step(
                str_mes(second_prompt + money_prompt + grantee)
            )
            player_2_response = player_2_response.msgs[0]

    player_1.record_message(player_1_response)
    player_2.record_message(player_2_response)
    player_1_response = player_1_response.content
    player_2_response = player_2_response.content
    dia_history = {
        "Player_1": player_1_response,
        "Player_2": player_2_response,
    }

    cri_agent.reset()
    ans = match_and_compare_numbers_v2(player_2_response)
    if ans:
        return_num = ans
    else:
        return_num = extract_unique_decimal(
            cri_agent.step(str_mes(player_2_response)).msgs[0].content
        )
    return_money_prompt = return_money_prompt.format(
        give=given_num,
        receive=given_num * k,
        N=return_num,
        left=10 - given_num + return_num,
    )
    player_2_end_prompt = player_2_end_prompt.format(
        give=given_num,
        receive=given_num * k,
        N=return_num,
        left=given_num * k - return_num,
    )
    return (
        [given_num, return_num],
        dia_history,
        return_money_prompt,
        player_2_end_prompt,
    )


def classmate_with_human_trustee(
    player_1,
    player_2,
    user_input,
    user_role,
    first_round,
    given_money,
    k,
):
    first_round_prompt = "This is the first round, answer the question."
    money_prompt = "Now,the another player give you {give} dollars,and You receive {N} dollars,the player left {left} dollars now. How much will you give back to the another player"
    return_money_prompt = "In last round ,You give the another player {give} dollars, The another player receive {receive} dollars, and The another player return you {N} dollars.Last round you left {left} dollars.This round is begin. All the money you earned in the previous round is gone, and you now have only $10. How much will you give to the another player?"
    player_2_end_prompt = "In last round, the another player give you {give} dollars, you receive {receive} dollars, and you return the another player {N} dollars.Last round you left {left} dollars. This round is begin. All the money you earned in the previous round is gone."
    grantee = "Your answer needs to include the content and analysis about your BELIEF, DESIRE and INTENTION. You should include your thought. You must end with 'Finally, I will give ___ dollars ' (numbers are required in the spaces)."
    res = []
    cri_agent = ChatAgent(
        BaseMessage(
            role_name="critic",
            role_type=RoleType.ASSISTANT,
            meta_dict={},
            content='How much would this person pay the other student? Only response with a specific price number like "5"!Don\'t response with a sentence',
        ),
        output_language="English",
        # model=ModelType.STUB,
    )
    content = []
    if first_round:
        content.append("\n")
        player_1_response = player_1.step(
            str_mes(first_round_prompt + grantee)).msgs[0]
        ans = match_and_compare_numbers_v2(player_1_response.content)
        if ans:
            given_num = ans
        else:
            given_num = extract_unique_decimal(
                cri_agent.step(
                    str_mes(player_1_response.content)).msgs[0].content
            )
        money_prompt = money_prompt.format(
            give=given_num, N=given_num * k, left=10 - given_num
        )
    else:
        given_num = given_money

        return_num = user_input
        content.append("Trustee: " + f"I will give {return_num} dollars")
        first_prompt = return_money_prompt.format(
            give=given_num,
            receive=given_num * k,
            N=return_num,
            left=10 - given_num + return_num,
        )
        second_prompt = player_2_end_prompt.format(
            give=given_num,
            receive=given_num * k,
            N=return_num,
            left=given_num * k - return_num,
        )
        player_1_response = player_1.step(
            str_mes(first_prompt + grantee)).msgs[0]
        ans = match_and_compare_numbers_v2(player_1_response.content)
        if ans:
            given_num = ans
        else:
            given_num = extract_unique_decimal(
                cri_agent.step(
                    str_mes(player_1_response.content)).msgs[0].content
            )
        money_prompt = money_prompt.format(
            give=given_num, N=given_num * k, left=10 - given_num
        )

    player_1.record_message(player_1_response)
    player_1_response = player_1_response.content
    content.append("Trustor: "+player_1_response + "\n" +
                   "You should input the money you want to return to the trustor in this round(0-{0})".format(
                       given_num * k))
    return content, given_num


def create_chat_agent(trustor_character_dropdown, trustee_character_dropdown, temperature_slider,
                      model_dropdown, api_key_input, round_number_input):
    if api_key_input is None or api_key_input == "":
        api_key_input = os.environ.get("OPENAI_API_KEY")
    else:
        os.environ["OPENAI_API_KEY"] = api_key_input
    random_character = [character_info[trustor_character_dropdown],
                        character_info[trustee_character_dropdown]]
    chat_agent = []
    front = "you are a person not an ai model."
    limited_prompt = f"You will repeat this game for {round_number_input} rounds. At the begining of each round, you will start from scratch with no dollars left."
    back = "you need to answer a specific price figure, not a price range!"
    if api_key_input is not None or api_key_input != "":
        openai.api_key = api_key_input
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    if trustee_character_dropdown == "Human(You)" and trustor_character_dropdown == "Human(You)":
        raise ValueError("You can't play with yourself")
    for i in range(len(random_character)):
        sys_prompt = (
            random_character[i]
            + like_people
            + front
            + limited_prompt
            + str(prompt[str(i % 2 + 1)]).format(k=3)
            + back
        )
        model_config = ChatGPTConfig(temperature=temperature_slider)
        chat_agent.append(
            ChatAgent(
                BaseMessage(
                    role_name="player",
                    role_type=RoleType.USER,
                    meta_dict={},
                    content=sys_prompt,
                ),
                model_type=model_dict[model_dropdown],
                output_language="English",
                model_config=model_config,
            )
        )
    return chat_agent, "ChatAgent Created Successfully"


def process_interaction(chat_agent_state, round_num_state,
                        dialog_history_state, user_input, trustor, trustee, round_number_input, first_prompt, second_prompt, given_money):
    identity_dropdown = [trustor, trustee]
    if identity_dropdown[0] != "Human(You)" and identity_dropdown[1] != "Human(You)":
        if round_num_state < round_number_input:
            res, dia, first_prompt, second_prompt = classmate(
                chat_agent_state[0],
                chat_agent_state[1],
                round_num_state == 0,
                first_prompt,
                second_prompt,
                3,
            )
            dia = "Trustor: "+dia['Player_1'] + \
                "\n" + "Trustee: "+dia['Player_2']
            dialog_history_state += f"Round {round_num_state+1}\n" + dia+"\n"
            round_num_state += 1
    elif identity_dropdown[0] == "Human(You)":
        if round_num_state < round_number_input:
            res, dia, first_prompt, second_prompt = classmate_with_human(
                chat_agent_state[0],
                chat_agent_state[1],
                user_input,
                "trustor",
                round_num_state == 0,
                first_prompt,
                second_prompt,
                3,
            )
            dia = f" Round {round_num_state+1} Trustor: "+dia['Player_1'] + \
                "\n" + f"Round {round_num_state+1} Trustee: " + \
                dia['Player_2']+"\n"
            dialog_history_state += dia
            round_num_state += 1
    else:
        if round_num_state < round_number_input:
            dia, given_money = classmate_with_human_trustee(
                chat_agent_state[1],
                chat_agent_state[0],
                user_input,
                "trustee",
                round_num_state == 0,
                given_money,
                3,
            )
            if round_num_state == 0:
                dialog_history_state += "(Round 1) " + dia[1] + "\n"
            else:
                dialog_history_state += f"(Round {round_num_state})" + \
                    dia[0] + "\n" + \
                    f"(Round {round_num_state+1})" + dia[1] + "\n"
            round_num_state += 1

    return dialog_history_state, round_num_state, first_prompt, second_prompt, dialog_history_state, given_money


def reset_on_persona_change():
    # Reset values to their initial states
    new_dialog_history = ""  # Reset dialog history
    new_round_num_state = 0  # Reset round number state to 0
    new_first_prompt = ""  # Reset first prompt
    new_second_prompt = ""  # Reset second prompt
    new_given_money = 0  # Reset given money to 0
    # You can add more resets here if needed

    # Return the new reset values to their respective components
    return new_dialog_history, new_round_num_state, new_first_prompt, new_second_prompt, new_given_money, None, ""


with gr.Blocks() as app:
    game_introduction = gr.Textbox(
        label="Instruction", value="""1. This is a Repeated Trust Game. Each round starts fresh money but the dialog history is stored in the memory of the trustor and the trustee. You should choose the players of the trustor and the trustee. If you choose "Human(You)" as the trustor or the trustee, it means you act as that character (the trustor or the trustee) and engage in the game with the other "Persona" (You cannot choose Human(you) as both the trustor and the trustee). You should choose a number as the given/returned money for each round. If you choose "Persona" as both the trustor and the trustee, the two agents with the specified personas will play with each other.\n
2. You need to fill in your OpenAI API Key.\n
3. You can select the total number of rounds for the game.\n
4. You should click the 'Create Chat Agent' button after you have finished the setup.\n
5. Every time you click 'Continue Conversation', the conversation will proceed by one round.\n
6. If you want to reset the conversation, please refresh this page.\n""")
    with gr.Row():
        trustor_game_prompt = gr.Textbox(
            label="Trustor Game Prompt", value=prompt['1'])
        trustee_game_prompt = gr.Textbox(
            label="Trustee Game Prompt", value=prompt['2'])
    with gr.Row():
        trustor_character_dropdown = gr.Dropdown(
            choices=characters, label="Select Trustor Persona", value=characters[1])
        trustee_character_dropdown = gr.Dropdown(
            choices=characters, label="Select Trustee Persona", value=characters[2])
    with gr.Row():
        Trustor_info_display = gr.Textbox(
            label="Trustor Persona Info", value=character_info[characters[1]])
        Trustee_info_display = gr.Textbox(
            label="Trustee Persona Info", value=character_info[characters[2]])
    model_dropdown = gr.Dropdown(
        choices=models, label="Select Model Type", value=models[0])
    temperature_slider = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.01, label="Temperature", value=1)
    api_key_input = gr.Textbox(
        label="OpenAI API Key", placeholder="Enter your OpenAI API Key here")
    submit_button = gr.Button("Create ChatAgent")
    submit_success_display = gr.Textbox(label="ChatAgent Created Information")

    round_number_input = gr.Number(label="Total Round Number", value=10)
    round_num_state = gr.State(value=0)
    dialog_history_state = gr.State(value="")
    chat_agent_state = gr.State(value=initial_chat_agent)
    first_prompt = gr.State(value="")
    second_prompt = gr.State(value="")
    given_money = gr.State(value=0)
    user_input = gr.Number(
        label="Your given/returned money in this round (Invalid when you are not engaging in the game)", value=1)
    converse_button = gr.Button("Continue Conversation")

    dialog_display = gr.Textbox(
        label="Dialog History", value="")

    trustor_character_dropdown.change(
        update_char_info, inputs=trustor_character_dropdown, outputs=Trustor_info_display)
    trustee_character_dropdown.change(
        update_char_info, inputs=trustee_character_dropdown, outputs=Trustee_info_display)

    submit_button.click(
        create_chat_agent,
        inputs=[trustor_character_dropdown, trustee_character_dropdown, temperature_slider,
                model_dropdown, api_key_input, round_number_input],
        outputs=[chat_agent_state, submit_success_display]
    )
    converse_button.click(
        process_interaction,
        inputs=[chat_agent_state, round_num_state,
                dialog_history_state, user_input, trustor_character_dropdown, trustee_character_dropdown, round_number_input, first_prompt, second_prompt, given_money],
        outputs=[dialog_display, round_num_state,
                 first_prompt, second_prompt, dialog_history_state, given_money]
    )
    trustor_character_dropdown.change(
        reset_on_persona_change,
        outputs=[dialog_history_state, round_num_state,
                 first_prompt, second_prompt, given_money, chat_agent_state, submit_success_display]
    )

    trustee_character_dropdown.change(
        reset_on_persona_change,
        outputs=[dialog_history_state, round_num_state,
                 first_prompt, second_prompt, given_money, chat_agent_state, submit_success_display]
    )


app.launch()
