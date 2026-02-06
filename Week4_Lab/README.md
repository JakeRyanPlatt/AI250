# Local Code Commenter / Explainer (Ollama + Python)

This project is a small command-line tool that sends a Python file to a local
Ollama model and returns:

- A plain-English explanation of what the code does
- Suggested improvements (bugs, style, performance, readability)

You can choose between two local models (example: `qwen2.5` and `llama3.2`) via a
command-line argument.

---

## Prerequisites

- Python 3.10+ (tested on Linux Ubuntu)
- `pip` (Python package manager)
- [Ollama](https://ollama.com) installed
- At least one of these models pulled with Ollama:
  - `qwen2.5`
  - `llama3.2`

### 1. Install and start Ollama

1. Install Ollama from the official website.
2. Start the Ollama server (it usually starts automatically):

   ```bash
   ollama serve
   ```
