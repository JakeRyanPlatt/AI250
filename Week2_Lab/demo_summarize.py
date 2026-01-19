import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

Paragraph = (
    "Artificial intelligence and cloud computing have become so tightly interwoven that "
    "it’s almost impossible to talk about one without acknowledging the influence of the other. "
    "Their relationship forms the backbone of today’s digital transformation, enabling "
    "organizations of every size to access capabilities that were once reserved for only "
    "the largest tech companies. At its core, AI thrives on data—massive volumes of it—and "
    "the cloud provides the elastic, on‑demand infrastructure required to store, process, "
    "and analyze that data at scale. Instead of investing in expensive hardware or "
    "maintaining complex data centers, businesses can tap into cloud platforms that "
    "automatically scale resources up or down based on workload demands. This flexibility "
    "allows AI models to be trained faster, deployed more efficiently, and updated "
    "continuously as new information becomes available. Cloud providers offer specialized "
    "hardware such as GPUs and TPUs, which dramatically accelerate machine learning tasks, "
    "making it feasible to experiment with deep learning architectures, natural language "
    "processing systems, and computer vision models without prohibitive costs. As a result, "
    "AI innovation has accelerated, and the barrier to entry has dropped significantly, "
    "empowering startups, researchers, and enterprises alike to build intelligent "
    "applications that would have been unimaginable a decade ago."
)

Prompt = f"Summarize the following paragraph into one to two sentences:\n\n{Paragraph}"


def timed(label, fn):
    start = time.perf_counter()

    try:
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "provider": label,
            "okay": True,
            "latency_ms": round(elapsed_ms, 1),
            "result": result,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "provider": label,
            "okay": False,
            "latency_ms": round(elapsed_ms, 1),
            "result": str(e),
        }


# Summarize paragraph using Hugging Face
def summarize_huggingface():
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "facebook/bart-large-cnn")

    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": Paragraph}

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Summarize the return
    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"]
    return data


def main():
    tests = []

    if os.getenv("HF_TOKEN"):
        tests.append(("Hugging Face Summarization", summarize_huggingface))
    else:
        print("Skipping Hugging Face (HF_TOKEN not set)")

    print("\nInput Paragraph:\n", Paragraph, "\n")

    results = [timed(label, fn) for label, fn in tests]

    for r in results:
        print(f"== {r['provider']} ==")
        print(f"OK: {r['okay']}  Latency: {r['latency_ms']} ms")
        if r["okay"]:
            print("Summary/Output:")
            if isinstance(r["result"], str):
                print(r["result"])
            else:
                print(json.dumps(r["result"], indent=2))
        else:
            print("Error:", r["result"])
        print()


if __name__ == "__main__":
    main()
