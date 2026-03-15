"""Post-processing for LLM tactical responses.

Removes artefacts that are useful for the model's reasoning but
distracting for a human operator, such as room coordinates and
numbered object labels.
"""
import re


def clean_response(text: str) -> str:
    # Remove "to (row,col)" / "at (row,col)" coordinate references
    text = re.sub(r'\s+(?:to|at)\s+\(\d+,\s*\d+\)', '', text)
    # Replace numbered labels: FakeVictim3 → a decoy, Victim11 → a victim
    text = re.sub(r'\bFakeVictim\d+\b', 'a decoy', text, flags=re.IGNORECASE)
    text = re.sub(r'\bVictim\d+\b', 'victims', text, flags=re.IGNORECASE)
    # Drop any remaining bare coordinates like (0,3)
    text = re.sub(r'\(\d+,\s*\d+\)', '', text)
    # Collapse extra spaces
    text = re.sub(r'  +', ' ', text).strip()
    return text
