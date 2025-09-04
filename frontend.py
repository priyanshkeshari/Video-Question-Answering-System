import streamlit as st
from PIL import Image
import os
import time
from typing import Any, Dict, Optional, Literal
from video_json import *  # contains detect_and_caption_objects, mark_motion_status
from langchain_integration import *
from app import frame_processing

# --- Setup ---
CWD = os.getcwd()
logo_name = "chatbot_image.jpg"
logo_path = os.path.join(CWD, logo_name)
logo_image = Image.open(logo_path)

folder_name = "frames"
for filename in os.listdir(folder_name):
        file_path_p = os.path.join(folder_name, filename)
        if os.path.isfile(file_path_p):
            os.remove(file_path_p)  # Remove file
        elif os.path.isdir(file_path_p):
            os.rmdir(file_path_p)
folder_path = os.path.join(CWD, folder_name)
os.makedirs(folder_path, exist_ok=True)

# --- Session state ---
if "video_summary" not in st.session_state:
    st.session_state.video_summary = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "queryres" not in st.session_state:
    st.session_state.queryres = {}
if "video_processing" not in st.session_state:
    st.session_state.video_processing = False
if "json_ready" not in st.session_state:
    st.session_state.json_ready = False  # Track JSON readiness


# --- Function with progress bar ---
def process_frames_and_caption_streamlit(folder_path, fps=1):
    frame_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith('.jpg')
    ])

    total_frames = len(frame_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_frames_data = []

    for idx, frame_path in enumerate(frame_files):
        frame_data = detect_and_caption_objects(frame_path)
        all_frames_data.append(frame_data)

        # Update progress
        progress_bar.progress((idx + 1) / total_frames)
        status_text.text(f"Processing frame {idx+1}/{total_frames}")

    # Mark motion status after all frames
    all_frames_data = mark_motion_status(all_frames_data, fps=fps)
    status_text.text("✅ Frame processing complete")
    return all_frames_data


# --- Header ---
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("## Personal Assistant")
with col2:
    st.image(logo_image, width=50)

st.divider()

# --- File uploader ---
last_file_name = None
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

query = st.text_input("Ask a question...")

# --- Process uploaded video ---
if uploaded_video and not st.session_state.video_summary and not st.session_state.video_processing:
    st.session_state.video_processing = True
    st.session_state.json_ready = False  # Reset JSON ready flag


    for filename in os.listdir('temp_video'):
        file_path_v = os.path.join('temp_video', filename)
        if os.path.isfile(file_path_v):
            os.remove(file_path_v)  # Remove file
        elif os.path.isdir(file_path_v):
            os.rmdir(file_path_v)

    os.makedirs("temp_video", exist_ok=True)
    video_path = os.path.join("temp_video", uploaded_video.name)


    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    if uploaded_video is not None:
        last_file_name = uploaded_video.name
        frame_processing(f'temp_video/{last_file_name}',1) 

    with st.spinner("Generating JSON for your video... Please wait"):
        fps = 1
        result = process_frames_and_caption_streamlit(folder_path, fps=fps)  # progress bar version
        temporal_result_dict = build_temporal_object_dict(result)
        pretty_output = temporal_result_dict

        # Generate video summary
        video_summary = generate_video_summary(pretty_output)
        st.session_state.video_summary["summary"] = video_summary["summary"]
        st.session_state.video_summary["events"] = video_summary["events"]

    st.session_state.video_processing = False
    st.session_state.json_ready = True
    st.success("Video processed and JSON generated successfully!")

# --- Button state logic ---
if not query:
    button_disabled = True  # Always disable if no question
elif uploaded_video:
    button_disabled = not st.session_state.json_ready  # Enable only if JSON ready
else:
    button_disabled = False  # No video, so general QA is fine


# --- Generate Answer Button ---
if st.button("Generate Answer", disabled=button_disabled):
    st.session_state.chat_history.append(("user", query))

    with st.spinner("Generating Answer... Please wait"):
        if st.session_state.video_summary:
            response = answer_from_summary(query, st.session_state.video_summary["summary"])
            st.session_state.chat_history.append(("ai_video_based", response))
        else:
            response = general_qa(query)
            st.session_state.chat_history.append(("ai_general_model", response))

    st.markdown("### 📌 Response:")

    if isinstance(response, dict):
        if response:
            st.header(f"📄")
            for ele in response.items():
                st.subheader(f"🎯 {ele[0]}")
                st.write(f"{ele[1]}")
        else:
            st.json(response)
    elif isinstance(response, str):
        st.markdown(f"**💬 Answer:** {response}")
    else:
        st.write(response)
