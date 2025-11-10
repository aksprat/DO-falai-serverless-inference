# app.py
import os
import time
import base64
import requests
from io import BytesIO
from flask import Flask, request, jsonify, send_file, Response

app = Flask(__name__)

# DigitalOcean Serverless Inference base + model access key (set this in env)
DO_INFERENCE_BASE = "https://inference.do-ai.run/v1/async-invoke"
DO_MODEL_ACCESS_KEY = os.getenv("DO_MODEL_ACCESS_KEY", "")
HEADERS = {"Authorization": f"Bearer {DO_MODEL_ACCESS_KEY}", "Content-Type": "application/json"}

# Polling config (seconds)
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.5"))
POLL_TIMEOUT = int(os.getenv("POLL_TIMEOUT", "60"))

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>fal text → image (DO) — PoC</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:0;background:#f3f6fb;display:flex;align-items:center;justify-content:center;height:100vh}
    .card{background:#fff;padding:24px;border-radius:12px;box-shadow:0 10px 30px rgba(20,30,60,.08);width:min(720px,92vw)}
    textarea{width:100%;min-height:120px;padding:12px;border-radius:8px;border:1px solid #e6eef8;resize:vertical}
    button{padding:10px 16px;border-radius:8px;border:0;background:#5561f2;color:#fff;cursor:pointer}
    .row{display:flex;gap:8px;margin-top:12px}
    img{max-width:100%;border-radius:10px;margin-top:12px;box-shadow:0 10px 30px rgba(0,0,0,.08)}
    .muted{color:#5b6b8a;font-size:0.9rem}
    .err{color:#b00020}
  </style>
</head>
<body>
  <div class="card">
    <h2>fal text → image (DigitalOcean)</h2>
    <p class="muted">Simple PoC — enter a prompt, click Generate, wait a few seconds.</p>

    <textarea id="prompt" placeholder="A cinematic photo of a futuristic city at sunset..."></textarea>

    <div class="row">
      <select id="model" style="flex:1;padding:8px;border-radius:8px;border:1px solid #e6eef8">
        <option value="fal-ai/flux/schnell">fal-ai/flux/schnell</option>
        <option value="fal-ai/fast-sdxl">fal-ai/fast-sdxl</option>
      </select>
      <button id="gen">🎨 Generate</button>
    </div>

    <div id="status" style="margin-top:12px"></div>
    <div id="result" style="margin-top:12px"></div>
  </div>

<script>
document.getElementById('gen').addEventListener('click', async () => {
  const prompt = document.getElementById('prompt').value.trim();
  const model = document.getElementById('model').value;
  const status = document.getElementById('status');
  const result = document.getElementById('result');
  result.innerHTML = '';
  if (!prompt) { status.innerHTML = '<div class="err">Please enter a prompt.</div>'; return; }
  status.textContent = 'Starting job...';
  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({prompt, model_id: model})
    });
    if (!resp.ok) {
      const text = await resp.text();
      status.innerHTML = '<div class="err">Error: ' + text + '</div>';
      return;
    }
    // The endpoint returns image bytes directly on success
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const img = document.createElement('img');
    img.src = url;
    result.appendChild(img);
    status.textContent = 'Done';
  } catch (err) {
    status.innerHTML = '<div class="err">Request failed: ' + err.message + '</div>';
  }
});
</script>
</body>
</html>
"""

# -------------------------
# Helpers for calling DO
# -------------------------
def start_async_invoke(model_id: str, input_payload: dict):
    body = {"model_id": model_id, "input": input_payload}
    r = requests.post(f"{DO_INFERENCE_BASE}/async-invoke", headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def get_status(request_id: str):
    r = requests.get(f"{DO_INFERENCE_BASE}/async-invoke/{request_id}/status", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def get_result(request_id: str):
    r = requests.get(f"{DO_INFERENCE_BASE}/async-invoke/{request_id}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()

def poll_until_complete(request_id: str, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL):
    start = time.time()
    while True:
        s = get_status(request_id)
        state = (s.get("status") or s.get("state") or "").upper()
        if state in ("COMPLETE", "SUCCEEDED", "SUCCESS"):
            return get_result(request_id)
        if state in ("FAILED", "ERROR"):
            raise RuntimeError(f"Job failed: {s}")
        if time.time() - start > timeout:
            raise TimeoutError("Polling timed out")
        time.sleep(interval)

def extract_image_bytes(result_json):
    # look for top-level url
    if isinstance(result_json, dict) and result_json.get("url"):
        url = result_json["url"]
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content, r.headers.get("Content-Type", "image/png")
    # look in output list
    output = result_json.get("output") or result_json.get("outputs") or result_json.get("results")
    if isinstance(output, list) and output:
        item = output[0]
        if isinstance(item, dict):
            if item.get("url"):
                r = requests.get(item["url"], timeout=30); r.raise_for_status()
                return r.content, r.headers.get("Content-Type", "image/png")
            if item.get("base64") or item.get("b64"):
                b64 = item.get("base64") or item.get("b64")
                return base64.b64decode(b64), "image/png"
        if isinstance(item, str) and item.startswith("http"):
            r = requests.get(item, timeout=30); r.raise_for_status()
            return r.content, r.headers.get("Content-Type", "image/png")
    # recursive search for first http string
    def find_url(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and v.startswith("http"):
                    return v
                found = find_url(v)
                if found: return found
        if isinstance(obj, list):
            for e in obj:
                found = find_url(e)
                if found: return found
        return None
    any_url = find_url(result_json)
    if any_url:
        r = requests.get(any_url, timeout=30); r.raise_for_status()
        return r.content, r.headers.get("Content-Type", "image/png")
    return None, None

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.route("/generate", methods=["POST"])
def generate():
    if not DO_MODEL_ACCESS_KEY:
        return "Server not configured with DO_MODEL_ACCESS_KEY", 500

    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    model_id = data.get("model_id") or "fal-ai/flux/schnell"

    if not prompt:
        return "Prompt is required", 400

    try:
        start = start_async_invoke(model_id, {"prompt": prompt})
        request_id = start.get("request_id") or start.get("id") or start.get("requestId")
        if not request_id:
            return jsonify({"error": "unexpected async-invoke response", "resp": start}), 502

        final = poll_until_complete(request_id)
        img_bytes, mime = extract_image_bytes(final)
        if not img_bytes:
            return jsonify({"error": "no image found in result", "result": final}), 502

        return send_file(BytesIO(img_bytes), mimetype=mime or "image/png")
    except TimeoutError as te:
        return f"Timed out: {te}", 504
    except requests.HTTPError as he:
        return f"HTTP error during inference: {he}", 502
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    # Helpful note on startup
    if not DO_MODEL_ACCESS_KEY:
        print("Warning: DO_MODEL_ACCESS_KEY not set. Set the env var before running.")
    app.run(host="0.0.0.0", port=8080)
