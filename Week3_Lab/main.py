import os
import traceback
import requests
from dotenv import load_dotenv
import PyPDF2

# Load HF_TOKEN from .env in this folder
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set. Add it to your .env file.")

# Hugging Face Inference API configuration
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # change if needed
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {hf_token}"}


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def generate_quiz_questions(text, num_questions=5):
    """Generate quiz questions from text using Hugging Face Inference API."""
    max_chars = 1500
    if len(text) > max_chars:
        text = text[:max_chars]

    prompt = f"""You are a helpful teaching assistant.

Create {num_questions} multiple choice quiz questions from this text.

Text:
{text}

Format each question exactly like this:

Question: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [correct letter]

Generate the questions now:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
        },
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Typical HF Inference output: [{"generated_text": "..."}]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception:
        print("Error generating questions (full traceback):")
        traceback.print_exc()
        return None


def save_quiz_to_file(questions, output_file="quiz_output.txt"):
    """Save generated quiz to a text file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== GENERATED QUIZ ===\n\n")
            f.write(questions)
        print(f"\nQuiz saved to {output_file}")
    except Exception as e:
        print(f"Error saving quiz: {e}")


def main():
    print("=" * 50)
    print("AI-Powered Quiz Generator")
    print("=" * 50)

    pdf_path = input("\nEnter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("Error: File not found!")
        return

    try:
        num_questions = int(
            input("How many questions would you like to generate? (1-10): ")
        )
        if num_questions < 1 or num_questions > 10:
            print("Using default: 5 questions")
            num_questions = 5
    except ValueError:
        print("Invalid input. Using default: 5 questions")
        num_questions = 5

    print("\nExtracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("Could not extract text from PDF.")
        return

    print(f"Extracted {len(text)} characters from PDF.")

    print(f"\nGenerating {num_questions} quiz questions using AI...")
    print("(This may take a moment...)")

    questions = generate_quiz_questions(text, num_questions)
    if questions:
        print("\n" + "=" * 50)
        print("GENERATED QUIZ")
        print("=" * 50)
        print(questions)
        print("=" * 50)

        save_option = input("\nWould you like to save the quiz to a file? (y/n): ") \
            .strip().lower()
        if save_option == "y":
            save_quiz_to_file(questions)
    else:
        print("Failed to generate quiz questions.")


if __name__ == "__main__":
    main()
