# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import re
from enum import Enum


class RoleType(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    CRITIC = "critic"
    EMBODIMENT = "embodiment"
    DEFAULT = "default"


class ModelType(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k-0613"
    GPT_4 = "gpt-4"
    GPT_4_32k = "gpt-4-32k"
    STUB = "stub"
    INSTRUCT_GPT = "text-davinci-003"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_3 = "davinci"
    VICUNA_7b = "vicuna-7b"
    VICUNA_16K = "vicuna-16k"
    VICUNA_13b = "vicuna-13b"
    VICUNA_33b = "vicuna-33b"
    LLAMA_2_7b = "llama-2-7b"
    LLAMA_2_13b = "llama-2-13b"
    LLAMA_2_70b = "llama-2-70b"

    @property
    def value_for_tiktoken(self) -> str:
        return self.value if self.name != "STUB" else "gpt-3.5-turbo"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model.

        Returns:
            bool: Whether this type of models belongs to OpenAI.
        """
        if self.name in {
            "GPT_3_5_TURBO",
            "GPT_3_5_TURBO_16K",
            "GPT_4",
            "GPT_4_32k",
            "INSTRUCT_GPT",
            "GPT_3_5_TURBO_INSTRUCT",
        }:
            return True
        else:
            return False

    @property
    def is_open_source(self) -> bool:
        r"""Returns whether this type of models is open-source.

        Returns:
            bool: Whether this type of models is open-source.
        """
        if self.name in {
            "LLAMA_2",
            "VICUNA",
            "VICUNA_16K",
            "VICUNA_7b",
            "VICUNA_13b",
            "VICUNA_33b",
            "LLAMA_2_7b",
            "LLAMA_2_13b",
            "LLAMA_2_70b",
        }:
            return True
        else:
            return False

    @property
    def token_limit(self) -> int:
        r"""Returns the maximum token limit for a given model.
        Returns:
            int: The maximum token limit for the given model.
        """
        if self is ModelType.GPT_3_5_TURBO or self is ModelType.GPT_3_5_TURBO_INSTRUCT:
            return 4096
        elif self is ModelType.GPT_3_5_TURBO_16K:
            return 16384
        elif self is ModelType.GPT_4:
            return 8192
        elif self is ModelType.GPT_4_32k:
            return 32768
        elif self is ModelType.STUB:
            return 4096
        elif (
            self is ModelType.LLAMA_2_7b
            or ModelType.LLAMA_2_13b
            or ModelType.LLAMA_2_70b
        ):
            return 4096
        elif (
            self is ModelType.VICUNA_7b or ModelType.VICUNA_13b or ModelType.VICUNA_33b
        ):
            # reference: https://lmsys.org/blog/2023-03-30-vicuna/
            return 2048
        elif self is ModelType.VICUNA_16K:
            return 16384
        else:
            raise ValueError("Unknown model type")

    def validate_model_name(self, model_name: str) -> bool:
        r"""Checks whether the model type and the model name matches.

        Args:
            model_name (str): The name of the model, e.g. "vicuna-7b-v1.5".
        Returns:
            bool: Whether the model type mathches the model name.
        """
        if self is ModelType.VICUNA_7b:
            pattern = r"^vicuna-\d+b-v\d+\.\d+$"
            return bool(re.match(pattern, model_name))
        elif self is ModelType.VICUNA_16K:
            pattern = r"^vicuna-\d+b-v\d+\.\d+-16k$"
            return bool(re.match(pattern, model_name))
        elif self is ModelType.LLAMA_2_7b:
            return self.value in model_name.lower() or "llama2" in model_name.lower()
        else:
            return self.value in model_name.lower()


class TaskType(Enum):
    AI_SOCIETY = "ai_society"
    CODE = "code"
    MISALIGNMENT = "misalignment"
    TRANSLATION = "translation"
    EVALUATION = "evaluation"
    SOLUTION_EXTRACTION = "solution_extraction"
    DEFAULT = "default"


__all__ = ["RoleType", "ModelType", "TaskType"]
