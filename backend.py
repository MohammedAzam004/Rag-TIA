import os
import re
import time
import warnings
import gc
from datetime import datetime
from typing import Callable, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from elevenlabs.client import ElevenLabs
# No longer using deprecated RetrievalQA - using manual RAG approach instead
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


def test_elevenlabs_api_key(api_key: str) -> tuple[bool, str]:
    """Test if an ElevenLabs API key is valid.
    
    Args:
        api_key: The API key to test
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not api_key or len(api_key) < 10:
        return False, "API key is too short"
    
    try:
        client = ElevenLabs(api_key=api_key.strip())
        # Try to get user info or voices to validate the key
        try:
            # Attempt a simple API call
            voices = list(client.voices.get_all())
            return True, f"✅ API key valid! Found {len(voices)} voices available."
        except AttributeError:
            # If voices.get_all() doesn't exist, try another approach
            return True, "✅ API key initialized (unable to verify voices)"
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in str(e) or "unauthorized" in error_msg:
            return False, "❌ API key rejected - Invalid or expired"
        if "403" in str(e) or "forbidden" in error_msg:
            return False, "❌ Access forbidden - Check account status"
        if "429" in str(e) or "rate limit" in error_msg:
            return False, "❌ Rate limit exceeded - Wait or upgrade plan"
        return False, f"❌ Error: {str(e)[:100]}"


def setup_rag_components(pdf_bytes: bytes):
    """Set up RAG components from PDF bytes."""
    if not pdf_bytes:
        print("❌ Error: No PDF data provided.")
        return None
    
    temp_file_path = "temp_uploaded_file.pdf"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(pdf_bytes)
    except IOError as e:
        print(f"❌ Error writing temporary PDF file: {e}")
        return None
    
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        if not documents:
            print("❌ Error: No content extracted from PDF.")
            return None
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            print("❌ Error: No text chunks created from PDF.")
            return None
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"✅ RAG components ready with {len(chunks)} chunks.")
        return vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as exc:
        import traceback
        print(f"❌ Error during RAG setup: {exc}")
        traceback.print_exc()
        return None
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"⚠️ Warning: Could not remove temporary file: {e}")


def _build_error_function(message: str) -> Callable[[], str]:
    def _error() -> str:
        return message
    return _error


def get_llm_pipeline(max_tokens: int = 128):
    """Get or initialize the LLM pipeline."""
    global llm_pipeline_cache
    if llm_pipeline_cache is None:
        model_name = "google/flan-t5-base"
        device = 0 if torch.cuda.is_available() else -1
        print(f"🔄 Loading language model: {model_name} on {'GPU' if device == 0 else 'CPU'}")
        try:
            generator = hf_pipeline(
                "text2text-generation",
                model=model_name,
                max_new_tokens=max_tokens,
                device=device,
            )
            llm_pipeline_cache = HuggingFacePipeline(pipeline=generator)
            print("✅ Language model loaded successfully.")
        except Exception as exc:
            print(f"❌ Failed to load LLM: {exc}")
            import traceback
            traceback.print_exc()
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
    """Generate text response with optional RAG context.
    
    Args:
        query: User's question or prompt
        retriever: Optional FAISS retriever for RAG context
        
    Returns:
        Generated text response
    """
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
        # Manual RAG implementation - more reliable than deprecated RetrievalQA
        if retriever:
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(final_query)
            
            if docs:
                # Combine document content as context
                context = "\n\n".join([doc.page_content for doc in docs[:3]])
                
                # Create enhanced prompt with context
                enhanced_prompt = f"""Context information:
{context}

Question: {final_query}

Answer based on the context above:"""
                
                # Generate with context
                outputs = llm_or_error.pipeline(
                    enhanced_prompt,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    max_new_tokens=128,
                )
            else:
                # No relevant docs found, use original query
                outputs = llm_or_error.pipeline(
                    final_query,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    max_new_tokens=128,
                )
        else:
            # No retriever - standard generation
            outputs = llm_or_error.pipeline(
                final_query,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                max_new_tokens=128,
            )
        
        # Extract result
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
    """Get or initialize the image generation pipeline."""
    global image_pipelines
    if model_id in image_pipelines and image_pipelines[model_id] is not None:
        return image_pipelines[model_id]

    print(f"🔄 Loading image model: {model_id}")
    
    # Clear memory before loading new model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device} with dtype: {torch_dtype}")
        
        # Load pipeline with safety checker disabled to avoid issues
        # Use low_cpu_mem_usage to reduce memory footprint
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True
        )
        pipe = pipe.to(device)
        
        # Enable memory-efficient attention if available
        try:
            pipe.enable_attention_slicing()
            print("✅ Attention slicing enabled for memory efficiency")
        except Exception:
            pass
        
        # Verify tokenizer is working
        try:
            test_tokens = pipe.tokenizer("test prompt", return_tensors="pt", max_length=77, truncation=True)
            print(f"✅ Tokenizer validated. Vocab size: {pipe.tokenizer.vocab_size}")
        except Exception as tok_err:
            print(f"⚠️ Tokenizer validation warning: {tok_err}")
        
        image_pipelines[model_id] = pipe
        print(f"✅ Image model {model_id} loaded successfully.")
    except MemoryError:
        print("❌ Out of memory while loading model")
        image_pipelines[model_id] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"❌ Error loading image pipeline: {exc}")
        import traceback
        traceback.print_exc()
        image_pipelines[model_id] = None
        gc.collect()
    return image_pipelines.get(model_id)


def _clean_image_prompt(prompt: str) -> str:
    """Clean and sanitize prompt for image generation.
    
    Removes problematic characters and ensures safe tokenization.
    """
    # Remove null bytes and control characters
    prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\t')
    
    # Replace multiple spaces with single space
    prompt = ' '.join(prompt.split())
    
    # Remove problematic unicode characters that might not be in vocab
    # Keep only ASCII printable and common punctuation
    cleaned = ''
    for char in prompt:
        if ord(char) < 128 or char.isalpha():  # ASCII or letters
            cleaned += char
        elif char in ' .,!?;:\'-()[]{}':  # Common punctuation
            cleaned += char
        else:
            cleaned += ' '  # Replace unknown chars with space
    
    return cleaned.strip()


def generate_image(prompt: str, model_id: str = "runwayml/stable-diffusion-v1-5") -> str:
    """Generate an image from a text prompt.
    
    Args:
        prompt: Text description of the image to generate
        model_id: Hugging Face model ID for the image generation model
        
    Returns:
        Path to the generated image file or error message
    """
    if not prompt or not prompt.strip():
        return "Error: Please provide a valid prompt."
    
    # Clean the prompt first
    prompt = _clean_image_prompt(prompt.strip())
    
    # Truncate prompt to avoid tokenization issues
    # CLIP tokenizer has 77 token limit, ~400 chars is safe
    if len(prompt) > 400:
        original_len = len(prompt)
        prompt = prompt[:400]
        print(f"⚠️ Prompt truncated from {original_len} to {len(prompt)} characters")
    
    pipe = get_image_pipeline(model_id)
    if pipe is None:
        return f"Error: Image model {model_id} could not be loaded."

    try:
        print(f"⏳ Generating image with {model_id}...")
        print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        start = time.time()
        
        # Free up memory before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate with explicit parameters to avoid tokenization errors
        # Reduced steps for faster generation and less chance of timeout
        output = pipe(
            prompt,
            num_inference_steps=30,  # Reduced from 50 for stability
            guidance_scale=7.5,
            max_length=77,  # Enforce CLIP token limit
        )
        
        if not output or not output.images:
            print("❌ No image generated")
            return "Error: Image generation returned no result."
        
        image = output.images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"generated_image_{timestamp}.png"
        
        try:
            image.save(image_path)
            elapsed = time.time() - start
            print(f"✅ Image saved to {image_path} ({elapsed:.2f}s)")
            
            # Clean up memory after generation
            del output
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return image_path
        except IOError as e:
            print(f"❌ Error saving image: {e}")
            return f"Error: Could not save image file. {e}"
    except KeyboardInterrupt:
        print("⚠️ Image generation interrupted by user")
        return "Error: Generation was interrupted."
    except RuntimeError as exc:
        error_msg = str(exc)
        print(f"❌ Runtime error generating image: {error_msg}")
        
        # Clean up on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if "out of bounds" in error_msg or "index" in error_msg:
            return "Error: Prompt contains invalid tokens. Try a simpler, shorter prompt."
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            return "Error: Out of memory. Try closing other applications or use a shorter prompt."
        return f"Error: {error_msg}"
    except MemoryError:
        print("❌ Out of memory error")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Error: Out of memory. Please try again with a simpler prompt."
    except Exception as exc:
        print(f"❌ Error generating image: {exc}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return f"Error: Could not generate image. {exc}"


def generate_audio(text_to_speak: str, api_key: str) -> str:
    """Generate audio from text using ElevenLabs API.
    
    Args:
        text_to_speak: Text to convert to speech
        api_key: ElevenLabs API key
        
    Returns:
        Path to generated audio file or error message
    """
    if not api_key or len(api_key) < 10:
        return "Error: Please enter a valid ElevenLabs API Key (minimum 10 characters)."
    
    if not text_to_speak or not text_to_speak.strip():
        return "Error: Please provide text for speech synthesis."
    
    # Trim text to reasonable length
    text_to_speak = text_to_speak.strip()
    if len(text_to_speak) > 5000:
        text_to_speak = text_to_speak[:5000]
        print(f"⚠️ Text trimmed to 5000 characters")
    
    try:
        print("🔄 Initializing ElevenLabs client...")
        print(f"📝 API Key length: {len(api_key)} chars")
        print(f"📝 Text length: {len(text_to_speak)} chars")
        
        # Initialize client with explicit API key
        client = ElevenLabs(api_key=api_key.strip())
        
        print("✅ ElevenLabs client initialized")
        print("⏳ Generating audio...")
        
        # Use the updated API format for v2.x
        # Try with default voice first
        try:
            audio_generator = client.text_to_speech.convert(
                voice_id="pNInz6obpgDQGcFmaJgB",  # Adam voice
                text=text_to_speak,
                model_id="eleven_multilingual_v2",
            )
            
            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)
            
        except Exception as voice_err:
            print(f"⚠️ Failed with Adam voice, trying Rachel voice: {voice_err}")
            # Try with Rachel voice as fallback
            audio_generator = client.text_to_speech.convert(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                text=text_to_speak,
                model_id="eleven_multilingual_v2",
            )
            audio_bytes = b"".join(audio_generator)
        
        if not audio_bytes or len(audio_bytes) < 100:
            return "Error: No valid audio data received from ElevenLabs. Check your account status."
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"generated_audio_{timestamp}.mp3"
        
        try:
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            print(f"✅ Audio saved to {audio_path} ({len(audio_bytes)} bytes)")
            return audio_path
        except IOError as e:
            print(f"❌ Error saving audio file: {e}")
            return f"Error: Could not save audio file. {e}"
            
    except AttributeError as attr_err:
        print(f"❌ API method error: {attr_err}")
        return f"Error: ElevenLabs API format mismatch. Try updating: pip install --upgrade elevenlabs"
    except ValueError as val_err:
        error_msg = str(val_err)
        print(f"❌ Value error: {error_msg}")
        if "api_key" in error_msg.lower():
            return "Error: Invalid API key format. Please check your key and try again."
        return f"Error: {error_msg}"
    except Exception as exc:
        print(f"❌ Error generating audio: {exc}")
        import traceback
        traceback.print_exc()
        
        error_msg = str(exc)
        error_lower = error_msg.lower()
        
        # More specific error handling
        if "401" in error_msg or "unauthorized" in error_lower or "authentication" in error_lower:
            return "Error: API key rejected by ElevenLabs. Possible reasons:\n- Invalid or expired API key\n- Free tier limit reached\n- Account suspended\n\nGet a new key at: https://elevenlabs.io/app/settings/api-keys"
        if "403" in error_msg or "forbidden" in error_lower:
            return "Error: Access forbidden. Your account may not have permission for this feature."
        if "429" in error_msg or "rate limit" in error_lower or "quota" in error_lower:
            return "Error: Rate limit or quota exceeded. Wait a few minutes or upgrade your plan."
        if "unusual activity" in error_lower or "suspicious" in error_lower:
            return "Error: ElevenLabs detected unusual activity. Contact their support or try a different account."
        if "connection" in error_lower or "timeout" in error_lower:
            return "Error: Network connection issue. Check your internet and try again."
        
        return f"Error: Could not generate audio - {error_msg}"
        return f"Error: Could not generate audio. {exc}"