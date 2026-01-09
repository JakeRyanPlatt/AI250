import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    # Load variables from .env into environment
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file.")

    # Configure OpenAI client (it will also read OPENAI_API_KEY from env)
    client = OpenAI(api_key=api_key)

    # Simple chat completion request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say one short sentence about Python."},
        ],
    )

    # Extract and print the AI-generated message
    message = response.choices[0].message.content
    print("AI response:")
    print(message)


if __name__ == "__main__":
    main()
