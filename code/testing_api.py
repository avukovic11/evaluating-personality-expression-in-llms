"""Quick OpenAI API key test. Run: python code/testing_api.py"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

api_key = os.environ["OPENAI_API_KEY"]
print(f"Loaded API key: {api_key}")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a pancake recipe in exactly 10 sentences.",
        }
    ],
    max_tokens=500,
)

print(response.choices[0].message.content)
