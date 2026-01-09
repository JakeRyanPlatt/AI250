from dotenv import load_dotenv 
load_dotenv()
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
model="gpt-5.2", input="What design decisions should be made when creating an app with django ontop of nginx"
)
print(response.output_text)
