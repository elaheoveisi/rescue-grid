from llama_index.core.llms import ChatMessage, MessageRole

from .parser import Goal, parse_goals
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
    prompt_type: str = "detailed",
    model: str = "gpt-5",
    provider: str = "openai",
) -> str:
    """Synchronously ask the LLM for advice given the current game observation.

    Args:
        obs: Enriched observation dict from the env.
        prompt_type: "sparse", "detailed", "semantic", or "decompose".
        model: Model name (e.g. "gpt-4o-mini", "gemini-1.5-flash").
        provider: "openai" or "gemini".

    Returns:
        The raw LLM response string.
    """
    llm = get_llm(model, provider)
    prompt = build_prompt(obs, prompt_type)
    print(prompt)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt),
    ]
    response = llm.chat(messages)
    print(response)
    print('-'*32)
    return response.message.content.strip()


def ask_goals(
    obs: dict,
    model: str = "gpt-5",
    provider: str = "openai",
) -> list[Goal]:
    """Ask the LLM to decompose the current mission into high-level goals.

    Uses the structured ``to_structured`` observation and the ``decompose``
    prompt.  Returns a parsed list of strategic :class:`~game.llm.parser.Goal`
    objects (e.g. ExploreRoom, RescueVictim, ClearDoor).

    Returns:
        Ordered list of :class:`~game.llm.parser.Goal` objects extracted from
        the ``<START>...<END>`` block.  Returns an empty list if the LLM
        response cannot be parsed.
    """
    raw = ask(obs, prompt_type="decompose", model=model, provider=provider)
    return parse_goals(raw, obs=obs)
