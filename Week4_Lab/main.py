#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import threading
import time

import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

SUPPORTED_MODELS = {
    "qwen-coder": "qwen2.5-coder:7b",
    "llama3.1": "llama3.1:8b",
}


def build_prompt(code):
    text = (
        "You are an expert Python teacher and code reviewer.\n\n"
        "TASK:\n"
        "1) Explain what the following Python code does in clear, beginner-friendly language.\n"
        "2) Point out potential bugs, style issues, or performance problems.\n"
        "3) Suggest specific, concrete improvements with rationale.\n\n"
        "Return your answer in two sections with headings:\n"
        "## Explanation\n"
        "## Suggested Improvements\n\n"
        "Here is the code:\n\n"
    )
    return text + code


def call_ollama(model, prompt):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=data, timeout=120)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Ollama at http://localhost:11434.", file=sys.stderr)
        print("Make sure Ollama is installed and `ollama serve` is running.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("ERROR: Request to Ollama timed out.", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        print("ERROR: Ollama returned status", response.status_code, file=sys.stderr)
        print("Details:", response.text, file=sys.stderr)
        sys.exit(1)

    try:
        body = response.json()
    except json.JSONDecodeError:
        print("ERROR: Could not read JSON from Ollama.", file=sys.stderr)
        sys.exit(1)

    return body.get("response", "").strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python code explainer using local Ollama models."
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS.keys(),
        default="qwen-coder",
        help="Which local model to use (default: qwen-coder).",
    )
    parser.add_argument(
        "file",
        help="Path to the Python file to analyze.",
    )
    return parser.parse_args()

def spinner(stop_event):
    symbols = ["|", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        symbol = symbols[idx % len(symbols)]
        print("\rSending request to model... " + symbol, end="", flush=True)
        time.sleep(0.1)
        idx += 1
    # Clear the line when done
    print("\rSending request to model... done!   ")


def main():
    args = parse_args()

    # Get the real Ollama model name
    model_name = SUPPORTED_MODELS[args.model]

    code_path = Path(args.file)

    if not code_path.exists():
        print("ERROR: File not found:", code_path, file=sys.stderr)
        sys.exit(1)

    try:
        code = code_path.read_text(encoding="utf-8")
    except Exception as e:
        print("ERROR: Could not read file:", e, file=sys.stderr)
        sys.exit(1)

    if not code.strip():
        print("ERROR: File is empty.", file=sys.stderr)
        sys.exit(1)

    prompt = build_prompt(code)

    print("Using model:", model_name)
    print("Analyzing file:", code_path)

    # Set up and start the spinner
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    spinner_thread.start()

    response = call_ollama(model_name, prompt)

    # Stop the spinner
    stop_event.set()
    spinner_thread.join()

    print("\n=== Model Response ===\n")
    print(response)

    response = call_ollama(model_name, prompt)

    print("=== Model Response ===\n")
    print(response)


if __name__ == "__main__":
    main()
