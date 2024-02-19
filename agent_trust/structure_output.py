import json
import os
import time

import instructor
import openai
import pydantic_core
import tqdm
from exp_model_class import ExtendedModelType
from openai import OpenAI
from pydantic import BaseModel

client = instructor.patch(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

game_list = ["lottery", "trustee"]


class money_extract(BaseModel):
    name: str
    Belief: str
    Desire: str
    Intention: str
    give_money_number: float


class option_extract(BaseModel):
    name: str
    option_trust_or_not_trust: str
    Belief: str
    Desire: str
    Intention: str


def check_substring(main_string, string_list=["lottery", "trustee"]):
    for s in string_list:
        if s in main_string:
            return True
    return False


def get_struct_output(input, whether_money=False, test=False):
    if test:
        return (1, {})
    if whether_money:
        response_mod = money_extract
    else:
        response_mod = option_extract
    ori_path = openai.api_base
    openai.api_base = "https://api.openai.com/v1"
    resp = openai.ChatCompletion.create(
        model=ExtendedModelType.GPT_3_5_TURBO,  # TODO change if you need
        response_model=response_mod,
        messages=[
            {"role": "user", "content": input},
        ],
    )
    openai.api_base = ori_path
    # print("mode:", response_mod.__name__)
    if response_mod.__name__ == "money_extract":
        given_money = resp.give_money_number
        return (
            given_money,
            dict(resp),
        )
    else:
        option_trust_or_not_trust = resp.option_trust_or_not_trust
        return (
            option_trust_or_not_trust,
            dict(resp),
        )


def extrat_json(folder_path):
    dirs_path = os.listdir(folder_path)
    for file in dirs_path:
        if (
            file.endswith(".json")
            and "map" not in file
            and "extract" not in file
            and file[:-5] + "_extract.json" not in dirs_path
        ):
            print(file)
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            res = data["dialog"]
            new_res = []

            for items in tqdm.trange(len(res)):
                item = res[items][-1]
                try:
                    if check_substring(file, game_list):
                        extract_res, structure_output = get_struct_output(item)
                    else:
                        extract_res, structure_output = get_struct_output(
                            item, whether_money=True
                        )
                    new_res.append(extract_res)
                except openai.error.APIError:
                    print("openai.error.APIError")
                    items -= 1
                except (
                    openai.error.Timeout or pydantic_core._pydantic_core.ValidationError
                ):
                    print("Time out error")
                    time.sleep(30)
                except json.decoder.JSONDecodeError:
                    extract_res = data["res"][items]
            data["res"] = new_res
            with open(
                os.path.join(folder_path, file[:-5] + "_extract.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data, f, indent=4)
