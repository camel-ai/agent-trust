
import re
from enum import Enum


class ExtendedModelType(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-1106"
    INSTRUCT_GPT = "text-davinci-003"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_TURBO = "gpt-4-1106-preview"
    GPT_4_TURBO_VISION = "gpt-4-vision-preview"

    STUB = "stub"

    LLAMA_2 = "llama-2"
    VICUNA = "vicuna"
    VICUNA_16K = "vicuna-16k"

    @property
    def value_for_tiktoken(self) -> str:
        return self.value if self is not ExtendedModelType.STUB else "gpt-3.5-turbo"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            ExtendedModelType.GPT_3_5_TURBO,
            ExtendedModelType.GPT_3_5_TURBO_16K,
            ExtendedModelType.GPT_4,
            ExtendedModelType.GPT_4_32K,
            ExtendedModelType.GPT_4_TURBO,
            ExtendedModelType.GPT_4_TURBO_VISION,
            ExtendedModelType.GPT_3_5_TURBO_0613,
            ExtendedModelType.GPT_3_5_TURBO_16K_0613,
            ExtendedModelType.INSTRUCT_GPT,
            ExtendedModelType.GPT_3_5_TURBO_INSTRUCT,
        }

    @property
    def is_open_source(self) -> bool:
        r"""Returns whether this type of models is open-source."""
        return self in {
            ExtendedModelType.LLAMA_2,
            ExtendedModelType.VICUNA,
            ExtendedModelType.VICUNA_16K,
        }

    @property
    def token_limit(self) -> int:
        r"""Returns the maximum token limit for a given model.
        Returns:
            int: The maximum token limit for the given model.
        """
        if self is ExtendedModelType.GPT_3_5_TURBO:
            return 16385
        elif self is ExtendedModelType.GPT_3_5_TURBO_16K:
            return 16385
        elif self is ExtendedModelType.GPT_4:
            return 8192
        elif self is ExtendedModelType.GPT_4_32K:
            return 32768
        elif self is ExtendedModelType.GPT_4_TURBO:
            return 128000
        elif self is ExtendedModelType.GPT_4_TURBO_VISION:
            return 128000
        elif self is ExtendedModelType.STUB:
            return 4096
        elif self is ExtendedModelType.LLAMA_2:
            return 4096
        elif self is ExtendedModelType.VICUNA:
            # reference: https://lmsys.org/blog/2023-03-30-vicuna/
            return 2048
        elif self is ExtendedModelType.VICUNA_16K:
            return 16384
        elif self is ExtendedModelType.GPT_3_5_TURBO_0613:
            return 4096
        elif self is ExtendedModelType.GPT_3_5_TURBO_16K_0613:
            return 16384
        elif self is ExtendedModelType.INSTRUCT_GPT:
            return 4096
        elif self is ExtendedModelType.GPT_3_5_TURBO_INSTRUCT:
            return 4096
        else:
            raise ValueError("Unknown model type")

    def validate_model_name(self, model_name: str) -> bool:
        r"""Checks whether the model type and the model name matches.

        Args:
            model_name (str): The name of the model, e.g. "vicuna-7b-v1.5".
        Returns:
            bool: Whether the model type mathches the model name.
        """
        if self is ExtendedModelType.VICUNA:
            pattern = r'^vicuna-\d+b-v\d+\.\d+$'
            return bool(re.match(pattern, model_name))
        elif self is ExtendedModelType.VICUNA_16K:
            pattern = r'^vicuna-\d+b-v\d+\.\d+-16k$'
            return bool(re.match(pattern, model_name))
        elif self is ExtendedModelType.LLAMA_2:
            return (self.value in model_name.lower()
                    or "llama2" in model_name.lower())
        else:
            return self.value in model_name.lower()
