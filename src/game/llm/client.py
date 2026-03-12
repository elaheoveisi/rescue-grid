from llama_index.core.llms import ChatMessage, MessageRole

from .prompts import detailed_prompt, sparse_prompt

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
    prompt_type: str = "detailed",
    model: str = "gpt-5",
    provider: str = "openai",
) -> str:
    """Synchronously ask the LLM for advice given the current game observation.

    Args:
        obs: Enriched observation dict from the env.
        prompt_type: "sparse" (action words only) or "detailed" (1-2 sentences).
        model: Model name (e.g. "gpt-4o-mini", "gemini-1.5-flash").
        provider: "openai" or "gemini".
    """

    prompt_fn = sparse_prompt if prompt_type == "sparse" else detailed_prompt
    llm = get_llm(model, provider)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt_fn(obs)),
        ChatMessage(role=MessageRole.USER, content="What should I do next?"),
    ]
    response = llm.chat(messages)
    return response.message.content.strip()
