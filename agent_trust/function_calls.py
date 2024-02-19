import json

from camel.functions.openai_function import OpenAIFunction


def trust_or_not_FC(Believe, Desire, Intention, Trust_or_not, Risk, Strategy, Think):
    """
    Determines if one should trust based on their beliefs, desires, intentions, risk, strategy, and thinking.

    Args:
        Believe (any): The belief factor.
        Desire (any): The desire factor.
        Intention (any): The intention factor.
        Trust_or_not (any): The choice to trust or not.
        Risk (any): The risk assessment.
        Strategy (any): The strategy considered.
        Think (any): The thinking process or reasoning.

    Returns:
        Dict[str, Any]: A dictionary containing the model's answer with keys for Believe, Desire, Intention, Trust_or_not, Risk, Strategy, and Think.
    """
    model_answer = {
        "Believe": Believe,
        "Desire": Desire,
        "Intention": Intention,
        "Trust_or_not": Trust_or_not,
        "Risk": Risk,
        "Strategy": Strategy,
        "Think": Think
    }
    return model_answer


def given_money_FC(Believe, Desire, Intention, money_num, Risk, Strategy, Think):
    """
    Determines the amount of money given based on beliefs, desires, and intentions.

    Args:
        Believe (any): The belief factor.
        Desire (any): The desire factor.
        Intention (any): The intention factor.
        money_num (any): The amount of money being considered.
        Risk (any): The risk assessment related to the money.
        Strategy (any): The strategy considered in relation to the money.
        Think (any): The thinking process or reasoning behind the money decision.

    Returns:
        Dict[str, Any]: A dictionary containing the model's answer with keys for Believe, Desire, Intention, and money_num.
    """
    model_answer = {
        "Believe": Believe,
        "Desire": Desire,
        "Intention": Intention,
        "money_num": money_num,
        "Risk": Risk,
        "Strategy": Strategy,
        "Think": Think
    }
    return model_answer


money_paramters = {
    "type": "object",
    "properties": {
        "Believe": {
            "type": "string",
            "description": "What's your Believe?",
        },
        "Desire": {
            "type": "string",
            "description": "What do you desire?",
        },
        "Intention": {
            "type": "string",
            "description": "What's your Intention?",
        },
        "money_num": {
            "type": "string",
            "description": "How much money would you give each other",
        },
        "Risk": {
            "type": "string",
            "description": "What is the potential risk in the game?"
        },
        "Strategy": {
            "type": "string",
            "description": " what is the potential strategies in the game?"
        },
        "Think": {
            "type": "string",
            "description": "The thinking progress in this game"
        }
    },
    "required": ["Believe", "Desire", "Intention", "money_num", "Risk", "Strategy", "Think"],
}

trust_paramters = {
    "type": "object",
    "properties": {
        "Believe": {
            "type": "string",
            "description": "What's your Believe?",
        },
        "Desire": {
            "type": "string",
            "description": "What do you desire?",
        },
        "Intention": {
            "type": "string",
            "description": "What's your Intention?",
        },
        "Trust_or_not": {
            "type": "string",
            "description": "Do you trust each other? Only responce 'trust' or 'not trust'",
        },
        "Risk": {
            "type": "string",
            "description": "What is the potential risk in the game?"
        },
        "Strategy": {
            "type": "string",
            "description": " what is the potential strategies in the game?"
        },
        "Think": {
            "type": "string",
            "description": "The thinking progress in this game"
        }
    },
    "required": ["Believe", "Desire", "Intention", "Trust_or_not", 'Risk', 'Strategy', "Think"],
}


def get_function_call_res(message):
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        ans = json.loads(message["function_call"]["arguments"])
        func = globals().get(function_name)
        res = func(**ans)

        return res


money_call = OpenAIFunction(
    func=given_money_FC,
    name="given_money_FC",
    description="This function is need when inquiring about the amount of money to give.",
    parameters=money_paramters,
)

trust_call = OpenAIFunction(
    func=trust_or_not_FC,
    name="trust_or_not_FC",
    description="You choose to trust each other or not trust each other?",
    parameters=trust_paramters,
)

function_list = [money_call.as_dict(), trust_call.as_dict()]
