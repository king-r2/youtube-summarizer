# app.py
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import requests
import re

st.set_page_config(page_title="YouTube → 3-Bullet Summary", layout="centered")

st.title("🎬 YouTube → 3-Bullet Summary")
st.write("Paste a YouTube URL and get a short 3-bullet summary using Hugging Face.")

# -----------------------------
# Helpers
# -----------------------------

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return ""

def get_transcript(video_id: str) -> str:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    texts = [t["text"] for t in transcript if t["text"].strip()]
    return " ".join(texts)

def hf_summarize(text, token, model="facebook/bart-large-cnn", max_length=120):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_length, "min_length": 20},
        "options": {"wait_for_model": True},
    }
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    data = res.json()

    if isinstance(data, list) and "summary_text" in data[0]:
        return data[0]["summary_text"]

    return str(data)

# -----------------------------
# UI
# -----------------------------

youtube_url = st.text_input("YouTube URL", placeholder="https://youtu.be/...")

if st.button("Summarize"):
    if not youtube_url:
        st.error("Please paste a YouTube URL.")
        st.stop()

    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Unable to extract video ID. Check the URL.")
        st.stop()

    # transcript
    with st.spinner("Fetching transcript..."):
        try:
            text = get_transcript(video_id)
        except TranscriptsDisabled:
            st.error("Transcript disabled for this video.")
            st.stop()
        except NoTranscriptFound:
            st.error("No transcript available.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Hugging Face token from Streamlit secrets
    token = st.secrets.get("HF_TOKEN")
    if not token:
        st.error("Missing Hugging Face token. Add it as HF_TOKEN in Streamlit secrets.")
        st.stop()

    # instruct summarizer to produce 3 bullets
    prompt = (
        "Summarize the following text into exactly 3 short bullet points:"
        "\n\n" + text
    )

    with st.spinner("Summarizing..."):
        summary = hf_summarize(prompt, token)

    # format into 3 bullets
    bullets = re.split(r"\n|•|-", summary)
    bullets = [b.strip() for b in bullets if b.strip()]

    # Ensure exactly 3 bullets
    if len(bullets) > 3:
        bullets = bullets[:3]
    elif len(bullets) < 3:
        while len(bullets) < 3:
            bullets.append("—")

    st.success("Summary:")
    for b in bullets:
        st.markdown(f"- {b}")
