import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")

# Config OpenAI client
client = OpenAI(api_key=api_key)

# Chat completion request
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": "Say one short sentence about python.",
        },
    ],
)

# Extract and print
message = response.choices[0].message.content
print("AI response:")
print(message)