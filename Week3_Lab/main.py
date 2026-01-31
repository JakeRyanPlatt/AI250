import os
import sys
import argparse
import traceback
from dotenv import load_dotenv
import PyPDF2
from huggingface_hub import InferenceClient

load_dotenv()


def get_hf_client():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in environment. Set it in .env or env vars.")
        return None
    try:
        return InferenceClient(api_key=token)
    except Exception:
        print("Failed to create Hugging Face client:")
        traceback.print_exc()
        return None


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_chunks = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
            return "\n\n".join(text_chunks)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def generate_quiz_questions(client, text, num_questions=5, max_chars=1500):
    """Generate quiz questions using the provided Hugging Face `InferenceClient`.

    Returns generated text or None on failure.
    """
    if client is None:
        print("No Hugging Face client available.")
        return None

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

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        # Depending on HF response shape; adapt defensively
        text_out = None
        try:
            text_out = completion.choices[0].message.content
        except Exception:
            try:
                # fallback for other shapes
                text_out = str(completion)
            except Exception:
                text_out = None
        return text_out
    except Exception:
        print("Error generating questions (full traceback):")
        traceback.print_exc()
        return None


def save_quiz_to_file(questions, output_file="quiz_output.txt"):
    """Save generated quiz to a text file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== GENERATED QUIZ ===\n\n")
            f.write(questions if questions else "")
        print(f"\nQuiz saved to {output_file}")
    except Exception as e:
        print(f"Error saving quiz: {e}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="AI-powered quiz generator from PDF using Hugging Face")
    p.add_argument("--pdf", help="Path to PDF file")
    p.add_argument("--num", type=int, default=5, help="Number of questions to generate (1-10)")
    p.add_argument("--output", help="Output file to save quiz (optional)")
    return p


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    pdf_path = args.pdf
    if not pdf_path:
        pdf_path = input("Enter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print("Error: File not found!")
        return 1

    num_questions = args.num
    if num_questions < 1 or num_questions > 10:
        print("Using default: 5 questions")
        num_questions = 5

    print("\nExtracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("Could not extract text from PDF.")
        return 1

    print(f"Extracted {len(text)} characters from PDF.")

    client = get_hf_client()
    if client is None:
        return 1

    print(f"\nGenerating {num_questions} quiz questions using AI...")
    print("(This may take a moment...)")

    questions = generate_quiz_questions(client, text, num_questions)
    if questions:
        print("\n" + "=" * 50)
        print("GENERATED QUIZ")
        print("=" * 50)
        print(questions)
        print("=" * 50)

        out_file = args.output
        if not out_file:
            save_option = input("\nWould you like to save the quiz to a file? (y/n): ").strip().lower()
            if save_option == "y":
                out_file = input("Enter output filename (default: quiz_output.txt): ").strip() or "quiz_output.txt"

        if out_file:
            save_quiz_to_file(questions, out_file)
        return 0
    else:
        print("Failed to generate quiz questions.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
