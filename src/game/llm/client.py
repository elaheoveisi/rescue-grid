from llama_index.core.llms import ChatMessage, MessageRole

from .parser import clean_response
from .prompts import build_prompt

llm_cache: dict = {}


def get_llm(model: str, provider: str):
    key = (provider, model)
    if key not in llm_cache:
        if provider == "openai":
            from llama_index.llms.openai import OpenAI

            llm_cache[key] = OpenAI(model=model)
        elif provider == "gemini":
            from llama_index.llms.google_genai import GoogleGenAI

            llm_cache[key] = GoogleGenAI(model=model)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Use 'openai' or 'gemini'."
            )
    return llm_cache[key]


def ask(
    obs: dict,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> str:
    llm = get_llm(model, provider)
    prompt = build_prompt(obs)
    print(prompt)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt),
    ]
    response = llm.chat(messages)
    print(response)
    print('-'*32)
    raw = response.message.content
    if "<START>" in raw and "<END>" in raw:
        extracted = raw.split("<START>")[1].split("<END>")[0].strip()
    else:
        extracted = raw.strip()
    return clean_response(extracted)


