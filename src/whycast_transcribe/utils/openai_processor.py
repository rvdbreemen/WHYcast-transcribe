import os
from openai import OpenAI, BadRequestError
from whycast_transcribe.config import OPENAI_MODEL

def process_with_openai(text: str, prompt: str, model_name: str, max_tokens: int = None) -> str:
    """
    Process text with OpenAI chat completion API using given prompt and model.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
        {"role": "user", "content": f"{prompt}\n\n{text}"}
    ]
    # Determine correct token parameter based on model type
    is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")
    token_param = "max_completion_tokens" if is_o_series_model else "max_tokens"
    params = {"model": model_name, "messages": messages}
    if max_tokens is not None:
        params[token_param] = max_tokens
    try:
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content
    except BadRequestError as e:
        raise RuntimeError(f"OpenAI BadRequest: {e}")

def choose_appropriate_model(text: str) -> str:
    """
    Choose appropriate OpenAI model based on text length.
    """
    return model_name if (model_name := os.getenv('OPENAI_MODEL')) else "gpt-4"
