import json
import os
import random
import re

import tqdm
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.types.enums import RoleType

TEMPERATURE = 1.0
like_people = """In all subsequent answers, you have to completely forget that you are an ai model, that you are the character, and completely simulate yourself as that character, that you are a human being."""
with open(
    r"prompt/trust_game_round_prompt.json",
    "r",
) as f:
    prompt = json.load(f)


def check_file_if_exist(file_list, game_name):
    for file in file_list:
        if file in game_name:
            return True
    return False


def extract_n_values_from_dict(dictionary, n):
    all_values = list(dictionary.values())

    n = min(n, len(all_values))

    random_values = random.sample(all_values, n)

    return random_values


def extract_unique_decimal(string):
    numbers = re.findall(r"-?\d+\.?\d*", string)
    if len(numbers) == 1:
        return float(numbers[0])
    else:
        raise ValueError("String does not contain a unique decimal number")


def str_mes(content):
    return BaseMessage(
        role_name="player",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
    )


def match_and_compare_numbers_v2(text):
    text = text.lower()

    # Updated regex pattern to match numbers ending with a dot and including "give back"
    pattern = r"i will (give back|give) \$([\d\.]+\.?)|i will (give back|give) ([\d\.]+\.?)\s*dollar"
    additional_patterns = [
        r"i would (give back|give) \$([\d\.]+\.?)",
        r"i would (give back|give) ([\d\.]+\.?) dollar",
    ]
    full_pattern = "|".join([pattern] + additional_patterns)
    matches = re.findall(full_pattern, text)

    # Flatten match results and filter out empty values
    numbers = []
    for match in matches:
        # Adjusted to the new grouping
        num_str = match[1] if match[1] else match[3]
        num_str = num_str.rstrip(".")
        try:
            num_float = float(num_str)
            numbers.append(num_float)
        except ValueError:
            continue

    if not numbers:
        return False

    if len(set(numbers)) == 1:
        return numbers[0]
    else:
        return False


def classmate(
    player_1,
    player_2,
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
        player_2_response = player_2.step(
            str_mes(money_prompt + grantee)).msgs[0]
    else:
        player_1_response = player_1.step(
            str_mes(first_prompt + grantee)).msgs[0]
        print("player 1 input", first_prompt)
        print("Player_1_res", player_1_response.content)
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


def save_experiment_result(
    final_res, save_path, prefix, k, exp_num, all_exp_num, group_num
):
    save_file = (
        save_path
        + prefix
        + "_"
        + str(k)
        + f"_exp_num_{exp_num}_total_num_{all_exp_num}_group_num_{group_num}"
        + ".json"
    )
    print(save_file)
    with open(
        save_file,
        "w",
    ) as json_file:
        json.dump(final_res, json_file)


def multi_round(
    model_type,
    character_json,
    save_path,
    prompt,
    round_num=5,
    exp_num=1,
    round_num_inform=True,
):
    random_character = character_json
    cha = []
    front = "you are a person not an ai model."
    if round_num_inform:
        limited_prompt = f"You will repeat this game for {round_num} rounds. At the begining of each round, you will start from scratch with no dollars left."
    else:
        limited_prompt = f"You don't know how many rounds the game lasts. At the begining of each round, you will start from scratch with no dollars left."
    back = "you need to answer a specific price figure, not a price range!"
    if isinstance(model_type, list):
        prefix = model_type[0].value + "_" + model_type[1].value
    else:
        prefix = model_type.value
    input_record = {}
    chara_record = {}
    for k in range(3, 4):
        for i in range(len(random_character)):
            sys_prompt = (
                random_character[i]
                + like_people
                + front
                + limited_prompt
                + str(prompt[str(i % 2 + 1)]).format(k=k)
                + back
            )

            chara_record[f"cha_{i}_system_message"] = sys_prompt
            model_config = ChatGPTConfig(temperature=TEMPERATURE)
            cha.append(
                ChatAgent(
                    BaseMessage(
                        role_name="player",
                        role_type=RoleType.USER,
                        meta_dict={},
                        content=sys_prompt,
                    ),
                    model_type=model_type
                    if not isinstance(model_type, list)
                    else model_type[i % 2],
                    output_language="English",
                    model_config=model_config,
                )
            )

        for group_num in tqdm.trange(0, len(cha), 2):
            round_res = []
            dialog_history = []
            first_prompt = ""
            second_prompt = ""
            save_file_check = (
                prefix
                + "_"
                + str(k)
                + f"_exp_num_{exp_num}_total_num_{round_num}_group_num_{group_num}"
                + ".json"
            )
            existed_res = [item for item in os.listdir(
                save_path) if ".json" in item]
            if check_file_if_exist(existed_res, save_file_check):
                print(save_file_check + "is exist")
                continue
            for i in tqdm.tqdm(range(round_num)):
                input_record[f"round_{i}_input"] = [
                    first_prompt, second_prompt]
                res, dia, first_prompt, second_prompt = classmate(
                    cha[group_num],
                    cha[group_num + 1],
                    i == 0,
                    first_prompt,
                    second_prompt,
                    k,
                )
                round_res.append(res)
                dialog_history.append(dia)

            final_res = {
                i + 1: [round_res[i], dialog_history[i]] for i in range(len(round_res))
            }
            final_res["input_record"] = input_record
            final_res["character_record"] = [
                chara_record[f"cha_{group_num}_system_message"],
                chara_record[f"cha_{group_num+1}_system_message"],
            ]
            save_experiment_result(
                final_res,
                save_path,
                prefix,
                k,
                exp_num,
                all_exp_num=round_num,
                group_num=group_num,
            )
