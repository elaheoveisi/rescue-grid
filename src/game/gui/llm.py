from openai import OpenAI

DUMMY = True  # Set to False to use the real OpenAI API


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


_SYSTEM_PROMPT = """\
You are an AI assistant embedded in a Search and Rescue (SAR) simulation.
The player controls an agent navigating a grid world to rescue victims.
Give concise tactical advice (2-3 sentences) based on the current game state.

Current game state:
{game_state}"""


def ask(game_state: str, model: str = "gpt-4o-mini") -> str:
    """Synchronously ask the LLM for advice given the current game state."""
    if DUMMY:
        return "Game state received. Keep searching for victims and avoid lava!"

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(game_state=game_state),
            },
            {
                "role": "user",
                "content": "What should I do next?",
            },
        ],
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()
