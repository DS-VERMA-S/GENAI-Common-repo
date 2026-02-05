import logging
import os
import time

from flask import Flask, jsonify, request

from app.model import ModelService

hf_model_name = "Qwen/Qwen3-0.6B"
device = "cpu"

logger = logging.getLogger("llm_inference_service")
if not logger.handlers:
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "llm_inference_service.log")
    log_format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def create_app() -> Flask:
    app = Flask(__name__)
    app.model_service = ModelService(hf_model_name, device)

    @app.get("/")
    def index():
        return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>LLM Inference UI</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      textarea { width: 100%; height: 140px; }
      .row { margin: 12px 0; }
      .label { display: inline-block; width: 140px; }
      pre { white-space: pre-wrap; background: #f5f5f5; padding: 12px; }
      button { padding: 8px 14px; }
    </style>
  </head>
  <body>
    <h2>LLM Inference UI</h2>
    <div class="row">
      <div class="label">Prompt</div>
      <textarea id="prompt"></textarea>
    </div>
    <div class="row">
      <span class="label">Max Tokens</span>
      <input id="max_tokens" type="number" value="128" />
    </div>
    <div class="row">
      <span class="label">Temperature</span>
      <input id="temperature" type="number" step="0.1" value="0.7" />
    </div>
    <div class="row">
      <button onclick="generate()">Generate</button>
    </div>
    <div class="row">
      <strong>Output</strong>
      <pre id="output"></pre>
    </div>
    <script>
      async function generate() {
        const prompt = document.getElementById("prompt").value;
        const max_tokens = Number(document.getElementById("max_tokens").value);
        const temperature = Number(document.getElementById("temperature").value);
        const output = document.getElementById("output");
        output.textContent = "Generating...";
        const res = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, max_tokens, temperature })
        });
        const data = await res.json();
        if (!res.ok) {
          output.textContent = data.detail || "Error";
          return;
        }
        output.textContent = data.output;
      }
    </script>
  </body>
</html>
"""

    @app.post("/generate")
    def generate():
        start = time.perf_counter()
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        max_tokens = int(data.get("max_tokens", 128))
        temperature = float(data.get("temperature", 0.7))
        if not prompt:
            return jsonify({"detail": "prompt is required"}), 400
        try:
            output = app.model_service.generate(prompt, max_tokens, temperature)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Flask /generate completed in %.2f ms (prompt_len=%d, max_tokens=%d, temperature=%.3f)",
                elapsed_ms,
                len(prompt),
                max_tokens,
                temperature,
            )
            return jsonify({"output": output})
        except Exception as e:
            logger.exception("Flask generation failed")
            return jsonify({"detail": str(e)}), 500

    return app


app = create_app()
