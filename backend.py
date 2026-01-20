import os
import re
import time
import warnings
from typing import Callable, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from elevenlabs.client import ElevenLabs
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline as hf_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

llm_pipeline_cache: Optional[Union[HuggingFacePipeline, Callable[[], str]]] = None
image_pipelines: dict[str, Optional[StableDiffusionPipeline]] = {}


def setup_rag_components(pdf_bytes: bytes):
    temp_file_path = "temp_uploaded_file.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(pdf_bytes)
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("✅ RAG components ready.")
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as exc:
        import traceback
        print(f"❌ Error during RAG setup: {exc}")
        traceback.print_exc()
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _build_error_function(message: str) -> Callable[[], str]:
    def _error() -> str:
        return message
    return _error


def get_llm_pipeline(max_tokens: int = 128):
    global llm_pipeline_cache
    if llm_pipeline_cache is None:
        model_name = "google/flan-t5-base"
        device = 0 if torch.cuda.is_available() else -1
        print(f"🔄 Loading language model: {model_name}")
        try:
            generator = hf_pipeline(
                "text2text-generation",
                model=model_name,
                max_new_tokens=max_tokens,
                device=device,
            )
            llm_pipeline_cache = HuggingFacePipeline(pipeline=generator)
        except Exception as exc:
            print(f"❌ Failed to load LLM: {exc}")
            llm_pipeline_cache = _build_error_function(f"Error loading language model: {exc}")
    return llm_pipeline_cache


def _clean_output(text: str, prompt: str) -> str:
    cleaned = text.strip()
    prompt_clean = prompt.strip().lower()
    if cleaned.lower().startswith(prompt_clean):
        cleaned = cleaned[len(prompt):].strip(" :\n\t")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned or text.strip()


def _reduce_repetitions(text: str) -> str:
    if not text:
        return text

    pattern = re.compile(r"((?:\b[\w'-]+\b[, ]+){1,6})(?:\1){2,}", re.IGNORECASE)
    while True:
        match = pattern.search(text)
        if not match:
            break
        phrase = match.group(1).strip(", ")
        text = text[: match.start()] + phrase + text[match.end():]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pruned = []
    seen = set()
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        key = re.sub(r"\s+", " ", stripped.lower())
        if key in seen:
            continue
        seen.add(key)
        pruned.append(stripped)
    if pruned:
        text = " ".join(pruned)

    words = text.split()
    if len(words) > 120:
        text = " ".join(words[:120])
    return text.strip()


def generate_text_response(query: str, retriever=None) -> str:
    llm_or_error = get_llm_pipeline()
    if not isinstance(llm_or_error, HuggingFacePipeline):
        return llm_or_error()

    final_query = query.strip()
    if not final_query:
        return "Please provide a prompt."

    small_greetings = {"hi", "hello", "hey", "hola", "hii", "hi!", "hello!"}
    if final_query.lower() in small_greetings:
        return "Hello! How can I assist you today?"

    try:
        if retriever:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_or_error,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
            )
            result = qa_chain.run(final_query)
        else:
            outputs = llm_or_error.pipeline(
                final_query,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                max_new_tokens=128,
            )
            if isinstance(outputs, list) and outputs:
                result = outputs[0].get("generated_text") or outputs[0].get("text", "")
            else:
                result = str(outputs)

        if isinstance(result, str):
            result = _clean_output(result, final_query)
            result = _reduce_repetitions(result)

        return result or "Model returned an empty response."
    except Exception as exc:
        print(f"❌ Error during text generation: {exc}")
        return f"Error: Could not generate response. Details: {exc}"


def get_image_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    global image_pipelines
    if model_id in image_pipelines:
        return image_pipelines[model_id]

    print(f"🔄 Loading image model: {model_id}")
    try:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        image_pipelines[model_id] = pipe
    except Exception as exc:
        print(f"❌ Error loading image pipeline: {exc}")
        image_pipelines[model_id] = None
    return image_pipelines.get(model_id)


def generate_image(prompt: str, model_id: str = "runwayml/stable-diffusion-v1-5") -> str:
    pipe = get_image_pipeline(model_id)
    if pipe is None:
        return f"Error: Image model {model_id} could not be loaded."

    try:
        print(f"⏳ Generating image with {model_id}...")
        start = time.time()
        output = pipe(prompt)
        image = output.images[0]
        image_path = "generated_image.png"
        image.save(image_path)
        print(f"✅ Image saved ({time.time() - start:.2f}s)")
        return image_path
    except Exception as exc:
        print(f"❌ Error generating image: {exc}")
        return f"Error: Could not generate image. {exc}"


def generate_audio(text_to_speak: str, api_key: str) -> str:
    if not api_key or len(api_key) < 10:
        return "Error: Please enter a valid ElevenLabs API Key."
    try:
        if not text_to_speak.strip():
            return "Error: Please provide text for speech synthesis."
        print("Initializing ElevenLabs client...")
        client = ElevenLabs(api_key=api_key)
        print("✅ ElevenLabs client ready.")
        print("⏳ Generating audio...")
        audio_stream = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            model_id="eleven_multilingual_v2",
            text=text_to_speak,
        )
        audio_bytes = b"".join(audio_stream)
        audio_path = "generated_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✅ Audio saved to {audio_path}")
        return audio_path
    except Exception as exc:
        print(f"❌ Error generating audio: {exc}")
        message = str(exc)
        if "401" in message:
            return "Error: Unauthorized. Check your ElevenLabs API key."
        if "Unusual activity detected" in message:
            return "Error: ElevenLabs blocked the request. Try again later."
        return f"Error: Could not generate audio. {exc}"