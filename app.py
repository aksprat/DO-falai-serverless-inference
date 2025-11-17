import os
import time
import requests
from io import BytesIO
from flask import Flask, request, send_file, Response

app = Flask(__name__)

# --- Configuration ---
DO_API_TOKEN = os.getenv("DO_API_TOKEN")  # set this in environment
DO_URL = "https://inference.do-ai.run/v1/async-invoke"
MODEL_ID = "fal-ai/fast-sdxl"

# Simple HTML UI
HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>fal-ai text-to-image (DigitalOcean)</title>
    <style>
      body { font-family: sans-serif; background: #f7f9fc; display: flex; flex-direction: column; align-items: center; padding-top: 50px; }
      textarea { width: 80%; height: 100px; padding: 10px; border-radius: 8px; border: 1px solid #ccc; font-size: 16px; }
      button { margin-top: 10px; padding: 10px 20px; background: #0069ff; color: white; border: none; border-radius: 6px; cursor: pointer; }
      img { margin-top: 20px; max-width: 80%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
      #status { margin-top: 15px; font-size: 14px; color: #555; }
    </style>
  </head>
  <body>
    <h2>Text → Image using fal-ai/fast-sdxl (DigitalOcean)</h2>
    <textarea id="prompt" placeholder="Enter your prompt here..."></textarea><br>
    <button onclick="generate()">Generate</button>
    <div id="status"></div>
    <img id="result" style="display:none;">
    <script>
      async function generate() {
        const prompt = document.getElementById('prompt').value.trim();
        const status = document.getElementById('status');
        const img = document.getElementById('result');
        if (!prompt) { status.textContent = 'Please enter a prompt.'; return; }
        status.textContent = 'Generating image... (this may take ~10s)';
        img.style.display = 'none';
        const resp = await fetch('/generate', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({prompt})
        });
        if (!resp.ok) {
          const text = await resp.text();
          status.textContent = 'Error: ' + text;
          return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        img.src = url;
        img.style.display = 'block';
        status.textContent = 'Done!';
      }
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return "Prompt required", 400

    headers = {
        "Authorization": f"Bearer {DO_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": MODEL_ID,
        "input": {"prompt": prompt}
    }

    try:
        # 1️⃣ Submit the job
        resp = requests.post(DO_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        job = resp.json()
        request_id = job.get("request_id") or job.get("id")
        if not request_id:
            return "Invalid response: no request_id", 500

        # 2️⃣ Poll for result
        for _ in range(30):
            status = requests.get(f"{DO_URL}/{request_id}/status", headers=headers, timeout=10).json()
            if status.get("status") in ("SUCCESS", "COMPLETE"):
                break
            time.sleep(2)

        # 3️⃣ Fetch result
        result = requests.get(f"{DO_URL}/{request_id}", headers=headers, timeout=30).json()
        img_url = result.get("output", [{}])[0].get("url") or result.get("url")

        if not img_url:
            return "No image URL in result", 500

        img_resp = requests.get(img_url, timeout=30)
        img_resp.raise_for_status()

        return send_file(BytesIO(img_resp.content), mimetype=img_resp.headers.get("Content-Type", "image/png"))

    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    if not DO_API_TOKEN:
        print("⚠️  Please set DO_API_TOKEN in your environment.")
    app.run(host="0.0.0.0", port=8080)
