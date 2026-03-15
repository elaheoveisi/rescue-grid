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
        elif provider == "google":
            from llama_index.llms.google_genai import GoogleGenAI

            llm_cache[key] = GoogleGenAI(model=model)
        else:
            raise ValueError(
                f"Unknown provider: {provider!r}. Use 'openai' or 'google'."
            )
    return llm_cache[key]


def ask(
    obs: dict,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> str:
    llm = get_llm(model, provider)
    prompt = build_prompt(obs)
    # Gemini requires at least one USER message; SYSTEM-only crashes with pop from empty list.
    # Sending prompt as USER works universally across providers.
    messages = [
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    response = llm.chat(messages)
    raw = response.message.content
    if raw is None:
        # Gemini returns blocks instead of a content string
        raw = "".join(
            b.text for b in (response.message.blocks or []) if hasattr(b, "text")
        )
    if "<START>" in raw and "<END>" in raw:
        extracted = raw.split("<START>")[1].split("<END>")[0].strip()
    else:
        extracted = raw.strip()
    return clean_response(extracted)
