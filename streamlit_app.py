# streamlit_app.py
"""
Very small PoC Streamlit app that sends a prompt to DigitalOcean fal model
via Serverless Inference and displays the generated image.

Requirements:
 - set environment variable DO_MODEL_ACCESS_KEY (model access key from DO console)
 - run: streamlit run streamlit_app.py
"""

import os
import time
import base64
import requests
import streamlit as st
from io import BytesIO

DO_INFERENCE_BASE = "https://inference.do-ai.run/v1"
DO_MODEL_ACCESS_KEY = os.getenv("DO_MODEL_ACCESS_KEY", "")
HEADERS = {"Authorization": f"Bearer {DO_MODEL_ACCESS_KEY}", "Content-Type": "application/json"}

# Polling defaults (tune if needed)
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.5"))
POLL_TIMEOUT = int(os.getenv("POLL_TIMEOUT", "60"))  # seconds

st.set_page_config(page_title="fal text → image (DO)", layout="centered")

st.title("fal text → image (DigitalOcean Serverless Inference)")
st.caption("Simple PoC — enter a prompt, press Generate, wait a few seconds.")

# Input area
prompt = st.text_area("Prompt", value="", height=120, placeholder="A cinematic photo of a futuristic city at sunset")
model_id = st.selectbox("Model (choose one)", options=["fal-ai/flux/schnell", "fal-ai/fast-sdxl"], index=0)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🎨 Generate")
with col2:
    st.write("")  # spacer

# Helper functions
def start_async_invoke(model_id: str, input_payload: dict):
    body = {"model_id": model_id, "input": input_payload}
    r = requests.post(f"{DO_INFERENCE_BASE}/async-invoke", headers=HEADERS, json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def get_status(request_id: str):
    r = requests.get(f"{DO_INFERENCE_BASE}/async-invoke/{request_id}/status", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()

def get_result(request_id: str):
    r = requests.get(f"{DO_INFERENCE_BASE}/async-invoke/{request_id}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()

def poll_until_complete(request_id: str, timeout_seconds=POLL_TIMEOUT, poll_interval=POLL_INTERVAL):
    start = time.time()
    while True:
        s = get_status(request_id)
        # status key may vary; check common names
        status_val = (s.get("status") or s.get("state") or "").upper()
        if status_val in ("COMPLETE", "SUCCEEDED", "SUCCESS"):
            return get_result(request_id)
        if status_val in ("FAILED", "ERROR"):
            raise RuntimeError(f"Inference job failed: {s}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for job (>{timeout_seconds}s)")
        time.sleep(poll_interval)

def extract_first_image_bytes(result_json):
    """
    Tries to find an image URL or base64 in result JSON and returns bytes.
    Returns (bytes, mime) or (None, None).
    """
    # 1) top-level url
    if isinstance(result_json, dict) and result_json.get("url"):
        url = result_json["url"]
        rr = requests.get(url, timeout=30)
        rr.raise_for_status()
        return rr.content, rr.headers.get("Content-Type", "image/png")

    # 2) output array
    output = result_json.get("output") or result_json.get("outputs") or result_json.get("results")
    if isinstance(output, list) and output:
        item = output[0]
        if isinstance(item, dict):
            if item.get("url"):
                rr = requests.get(item["url"], timeout=30)
                rr.raise_for_status()
                return rr.content, rr.headers.get("Content-Type", "image/png")
            if item.get("base64") or item.get("b64"):
                b64 = item.get("base64") or item.get("b64")
                return base64.b64decode(b64), "image/png"
            # sometimes the item itself is a string URL
            if isinstance(item, str) and item.startswith("http"):
                rr = requests.get(item, timeout=30)
                rr.raise_for_status()
                return rr.content, rr.headers.get("Content-Type", "image/png")

    # 3) search recursively for a url string
    def find_url(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and v.startswith("http"):
                    return v
                found = find_url(v)
                if found:
                    return found
        if isinstance(obj, list):
            for el in obj:
                found = find_url(el)
                if found:
                    return found
        return None

    u = find_url(result_json)
    if u:
        rr = requests.get(u, timeout=30)
        rr.raise_for_status()
        return rr.content, rr.headers.get("Content-Type", "image/png")

    return None, None

# UI: run job
if generate_btn:
    if not DO_MODEL_ACCESS_KEY:
        st.error("DO_MODEL_ACCESS_KEY is not set. Set environment variable and restart.")
    elif not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        status_placeholder = st.empty()
        img_placeholder = st.empty()
        status_placeholder.info("Starting job...")

        try:
            # Start job
            start_resp = start_async_invoke(model_id, {"prompt": prompt})
            request_id = start_resp.get("request_id") or start_resp.get("id") or start_resp.get("requestId")
            if not request_id:
                st.error("Unexpected response from async-invoke: " + str(start_resp))
            else:
                status_placeholder.info(f"Job started: {request_id}. Polling for result...")
                # Poll
                final = poll_until_complete(request_id)
                status_placeholder.success("Job complete — retrieving image...")

                img_bytes, mime = extract_first_image_bytes(final)
                if not img_bytes:
                    st.error("No image found in model result. See raw result:")
                    st.json(final)
                else:
                    # Display the image (Streamlit accepts bytes or PIL)
                    img_placeholder.image(img_bytes, use_column_width=True)
                    # Optionally show raw JSON for debugging
                    with st.expander("Show raw model result (for debugging)"):
                        st.json(final)
        except TimeoutError as te:
            status_placeholder.error(str(te))
        except requests.HTTPError as he:
            status_placeholder.error(f"HTTP error: {he}")
        except Exception as e:
            status_placeholder.error(f"Error: {e}")
