#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const http = require("http");

const OLLAMA_HOST = "localhost";
const OLLAMA_PORT = 11434;
const OLLAMA_PATH = "/api/generate";

const SUPPORTED_MODELS = {
  "qwen-coder": "qwen2.5-coder:7b",
  "llama3.1": "llama3.1:8b",
};

function parseArgs() {
  const args = process.argv.slice(2);
  let modelKey = "qwen-coder";
  let filePath = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--model" && i + 1 < args.length) {
      modelKey = args[i + 1];
      i++;
    } else if (!filePath) {
      filePath = args[i];
    }
  }

  return { modelKey, filePath };
}

function buildPrompt(code) {
  return (
    "You are an expert Python teacher and code reviewer.\n\n" +
    "TASK:\n" +
    "1) Explain what the following Python code does in clear, beginner-friendly language.\n" +
    "2) Point out potential bugs, style issues, or performance problems.\n" +
    "3) Suggest specific, concrete improvements with rationale.\n\n" +
    "Return your answer in two sections with headings:\n" +
    "## Explanation\n" +
    "## Suggested Improvements\n\n" +
    "Here is the code:\n\n" +
    code
  );
}

function startSpinner() {
  const symbols = ["|", "/", "-", "\\"];
  let idx = 0;
  process.stdout.write("Sending request to model... ");
  const id = setInterval(() => {
    const s = symbols[idx % symbols.length];
    process.stdout.write("\rSending request to model... " + s);
    idx++;
  }, 100);
  return id;
}

function stopSpinner(id) {
  clearInterval(id);
  process.stdout.write("\rSending request to model... done!   \n");
}

function callOllama(modelName, prompt, callback) {
  const body = JSON.stringify({
    model: modelName,
    prompt: prompt,
    stream: false,
  });

  const options = {
    hostname: OLLAMA_HOST,
    port: OLLAMA_PORT,
    path: OLLAMA_PATH,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body),
    },
  };

  const req = http.request(options, (res) => {
    let data = "";

    res.on("data", (chunk) => {
      data += chunk;
    });

    res.on("end", () => {
      if (res.statusCode !== 200) {
        return callback(
          new Error("Ollama returned status " + res.statusCode + ": " + data)
        );
      }
      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch (e) {
        return callback(new Error("Could not parse JSON from Ollama"));
      }
      callback(null, (parsed.response || "").trim());
    });
  });

  req.on("error", (err) => {
    callback(err);
  });

  req.write(body);
  req.end();
}

function main() {
  const { modelKey, filePath } = parseArgs();

  if (!filePath) {
    console.error("Usage: node main.js [--model MODEL] PATH_TO_PY_FILE");
    console.error("Models:", Object.keys(SUPPORTED_MODELS).join(", "));
    process.exit(1);
  }

  if (!SUPPORTED_MODELS[modelKey]) {
    console.error("Unknown model:", modelKey);
    console.error("Supported models are:", Object.keys(SUPPORTED_MODELS).join(", "));
    process.exit(1);
  }

  const modelName = SUPPORTED_MODELS[modelKey];
  const absPath = path.resolve(filePath);

  if (!fs.existsSync(absPath)) {
    console.error("File not found:", absPath);
    process.exit(1);
  }

  let code;
  try {
    code = fs.readFileSync(absPath, "utf8");
  } catch (e) {
    console.error("Could not read file:", e.message);
    process.exit(1);
  }

  if (!code.trim()) {
    console.error("File is empty.");
    process.exit(1);
  }

  const prompt = buildPrompt(code);

  console.log("Using model:", modelName);
  console.log("Analyzing file:", absPath);

  const spinnerId = startSpinner();

  callOllama(modelName, prompt, (err, response) => {
    stopSpinner(spinnerId);

    if (err) {
      console.error("Error talking to Ollama:", err.message);
      process.exit(1);
    }

    console.log("\n=== Model Response ===\n");
    console.log(response);
  });
}

main()
