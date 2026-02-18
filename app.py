import os
import time
import streamlit as st
from backend import setup_rag_components, generate_text_response, generate_image, generate_audio, test_elevenlabs_api_key

st.set_page_config(page_title="Multi Tasker AI", layout="wide", page_icon="🚀")

if "app_loaded" not in st.session_state:
    st.session_state.app_loaded = False
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Text"
if "text_messages" not in st.session_state:
    st.session_state.text_messages = []
if "image_messages" not in st.session_state:
    st.session_state.image_messages = []
if "audio_messages" not in st.session_state:
    st.session_state.audio_messages = []
if "rag_retriever" not in st.session_state:
    st.session_state.rag_retriever = None
if "text_uploaded_file_name" not in st.session_state:
    st.session_state.text_uploaded_file_name = None
if "selected_image_model" not in st.session_state:
    st.session_state.selected_image_model = "runwayml/stable-diffusion-v1-5"
if "elevenlabs_key" not in st.session_state:
    st.session_state.elevenlabs_key = ""
if "image_prompt_buffer" not in st.session_state:
    st.session_state.image_prompt_buffer = ""
if "audio_prompt_buffer" not in st.session_state:
    st.session_state.audio_prompt_buffer = ""

@st.cache_resource(show_spinner="Processing PDF...")
def load_rag_retriever(file_bytes: bytes, file_name: str):
    return setup_rag_components(file_bytes)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-bottom: 8rem;
    }
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 1.1rem;
        background: transparent;
        padding: 0;
        margin-top: 1.5rem;
    }
    [data-testid="stChatInput"] > div {
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(17, 19, 28, 0.95);
        border-radius: 20px;
        padding: 0.8rem 1rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
    }
    [data-testid="stChatInputTextarea"] textarea {
        min-height: 1.2rem;
        font-size: 0.95rem;
        border: none !important;
        background: transparent !important;
        color: #fff !important;
    }
    [data-testid="stChatInputTextarea"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.7);
    }
    [data-testid="stChatInputSubmitButton"] button {
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.app_loaded:
    st.title("🚀 Multi Tasker AI")
    st.markdown("Your multitool for Text, Image, and Audio generation.")
    with st.spinner("Initializing..."):
        time.sleep(1.2)
    st.session_state.app_loaded = True
    st.rerun()

mode_labels = {
    "📝 Text Generation": "Text",
    "🎨 Image Generation": "Image",
    "🎙️ Audio Generation": "Audio",
}

with st.sidebar:
    st.title("Multi Tasker AI")
    st.divider()
    selected_mode_display = st.radio(
        "Select Mode",
        list(mode_labels.keys()),
        index=list(mode_labels.values()).index(st.session_state.current_mode),
        label_visibility="collapsed",
    )
    st.session_state.current_mode = mode_labels[selected_mode_display]
    st.divider()

    if st.session_state.current_mode == "Text":
        st.header("Document Context (Optional)")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            key="text_pdf_uploader_sidebar",
            help="Attach a PDF to enable retrieval-augmented responses.",
        )
        if uploaded_file is not None:
            if st.session_state.text_uploaded_file_name != uploaded_file.name:
                retriever = load_rag_retriever(uploaded_file.getvalue(), uploaded_file.name)
                if retriever:
                    st.session_state.rag_retriever = retriever
                    st.session_state.text_uploaded_file_name = uploaded_file.name
                    st.success(f"RAG Active: {uploaded_file.name}")
                else:
                    st.session_state.rag_retriever = None
                    st.session_state.text_uploaded_file_name = None
                    st.error("Failed to build RAG retriever.")
        elif st.session_state.text_uploaded_file_name:
            st.session_state.rag_retriever = None
            st.session_state.text_uploaded_file_name = None
            load_rag_retriever.clear()
            st.info("RAG context cleared.")

        st.divider()
        st.header("Model")
        st.info("Using Model: google/flan-t5-base")

    elif st.session_state.current_mode == "Image":
        st.header("Model Selection")
        models = ["runwayml/stable-diffusion-v1-5", "stabilityai/sdxl-turbo"]
        st.session_state.selected_image_model = st.selectbox(
            "Choose Image Model",
            models,
            index=models.index(st.session_state.selected_image_model),
            help="SDXL Turbo is faster; Stable Diffusion v1.5 is higher quality.",
        )

    elif st.session_state.current_mode == "Audio":
        st.header("ElevenLabs API Key")
        st.markdown("Get your API key at: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.elevenlabs_key = st.text_input(
                "API Key",
                type="password",
                placeholder="Enter ElevenLabs key (e.g., sk_...)",
                value=st.session_state.elevenlabs_key,
            )
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🔍 Test Key", help="Verify your API key is valid"):
                if st.session_state.elevenlabs_key:
                    with st.spinner("Testing API key..."):
                        is_valid, message = test_elevenlabs_api_key(st.session_state.elevenlabs_key)
                    if is_valid:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter an API key first")
        
        if st.session_state.elevenlabs_key:
            st.info(f"API Key: {'*' * (len(st.session_state.elevenlabs_key) - 4)}{st.session_state.elevenlabs_key[-4:]}")
        else:
            st.warning("⚠️ No API key entered. Please add your ElevenLabs API key above.")

st.header(f"{selected_mode_display} Interface")

if st.session_state.current_mode == "Text":
    st.caption("Using Model: google/flan-t5-base")
    if not st.session_state.text_messages:
        st.info("Ask anything to get started.")
    for msg in st.session_state.text_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask anything…")
    if user_prompt is not None:
        prompt = user_prompt.strip()
        if prompt:
            st.session_state.text_messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                reply = generate_text_response(prompt, retriever=st.session_state.rag_retriever)
            st.session_state.text_messages.append({"role": "assistant", "content": reply})
            st.rerun()
        else:
            st.warning("Please enter a prompt.")

elif st.session_state.current_mode == "Image":
    st.caption(f"Using Model: {st.session_state.selected_image_model}")
    for msg in st.session_state.image_messages:
        with st.chat_message(msg["role"]):
            if msg.get("kind") == "image" and os.path.exists(msg["content"]):
                st.image(msg["content"])
            else:
                st.markdown(msg["content"])

    with st.form("image_prompt_form", clear_on_submit=False):
        st.session_state.image_prompt_buffer = st.text_area(
            "Image Prompt",
            value=st.session_state.image_prompt_buffer,
            placeholder="Describe the image you want to generate...",
        )
        submitted = st.form_submit_button("Generate Image", type="primary")

    if submitted:
        prompt = st.session_state.image_prompt_buffer.strip()
        if not prompt:
            st.warning("Please enter an image prompt before submitting.")
        else:
            st.session_state.image_messages.append({"role": "user", "content": prompt})
            try:
                with st.spinner("Generating image... This may take 30-60 seconds."):
                    image_path = generate_image(prompt, st.session_state.selected_image_model)
                
                if image_path and os.path.exists(image_path):
                    st.session_state.image_messages.append(
                        {"role": "assistant", "content": image_path, "kind": "image"}
                    )
                    st.success("Image generated successfully!")
                else:
                    st.session_state.image_messages.append({"role": "assistant", "content": image_path or "Error: No image path returned."})
                    st.error("Image generation failed. Please try again.")
            except Exception as e:
                error_msg = f"Error: Generation failed - {str(e)}"
                st.session_state.image_messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
            finally:
                st.session_state.image_prompt_buffer = ""
                st.rerun()

elif st.session_state.current_mode == "Audio":
    st.caption("Using Model: ElevenLabs")
    for msg in st.session_state.audio_messages:
        with st.chat_message(msg["role"]):
            if msg.get("kind") == "audio" and os.path.exists(msg["content"]):
                st.audio(msg["content"])
            else:
                st.markdown(msg["content"])

    with st.form("audio_prompt_form", clear_on_submit=False):
        st.session_state.audio_prompt_buffer = st.text_area(
            "Speech Prompt",
            value=st.session_state.audio_prompt_buffer,
            placeholder="Enter text to generate audio...",
        )
        submitted = st.form_submit_button("Generate Audio", type="primary")

    if submitted:
        prompt = st.session_state.audio_prompt_buffer.strip()
        if not prompt:
            st.warning("Please enter text before submitting.")
        elif not st.session_state.elevenlabs_key:
            st.error("⚠️ Please set your ElevenLabs API key in the sidebar first.")
        elif len(st.session_state.elevenlabs_key) < 10:
            st.error("⚠️ API key appears invalid (too short). Please check your key.")
        else:
            st.session_state.audio_messages.append({"role": "user", "content": prompt})
            try:
                with st.spinner("🎙️ Generating audio... This may take 10-30 seconds."):
                    audio_path = generate_audio(prompt, st.session_state.elevenlabs_key)
                
                if audio_path and os.path.exists(audio_path):
                    st.session_state.audio_messages.append(
                        {"role": "assistant", "content": audio_path, "kind": "audio"}
                    )
                    st.success("✅ Audio generated successfully!")
                else:
                    # Error message from backend
                    st.session_state.audio_messages.append({"role": "assistant", "content": audio_path or "Error: No audio generated."})
                    if "unauthorized" in audio_path.lower() or "401" in audio_path:
                        st.error("❌ API Key Error: Your key was rejected. Try:\n1. Generate a new key at https://elevenlabs.io/app/settings/api-keys\n2. Test it using the 🔍 Test Key button\n3. Make sure you have credits in your account")
                    else:
                        st.error(f"❌ {audio_path}")
            except Exception as e:
                error_msg = f"Error: Audio generation failed - {str(e)}"
                st.session_state.audio_messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
            finally:
                st.session_state.audio_prompt_buffer = ""
                st.rerun()
                st.session_state.audio_messages.append({"role": "assistant", "content": audio_path})
            st.session_state.audio_prompt_buffer = ""
            st.rerun()