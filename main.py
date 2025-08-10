from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import asyncio
import httpx
import os

app = FastAPI(title="Ollama-compatible API Proxy", version="1.0.0")

# Enable CORS for mobile app compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - set these as environment variables in your HF Space
LLM_SPACE_URL = os.getenv("LLM_SPACE_URL", "https://amd-gpt-oss-120b-chatbot.hf.space")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for private spaces

# Pydantic models matching Ollama API
class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150

class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150

def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string"""
    prompt_parts = []
    
    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"<|system|>\n{message.content}")
        elif message.role == "user":
            prompt_parts.append(f"<|user|>\n{message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{message.content}")
    
    # Add the assistant prompt at the end
    prompt_parts.append("<|assistant|>\n")
    
    return "\n".join(prompt_parts)

async def stream_from_llm_space(prompt: str, temperature: float = 0.7, max_tokens: int = 150):
    """Stream responses from the LLM Space and convert to Ollama format"""
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add authorization header if HF token is provided
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    # Payload for the LLM space (adjust based on your LLM space's API)
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "do_sample": True,
            "stream": True
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Try different common endpoints for HF Spaces
            endpoints_to_try = [
                f"{LLM_SPACE_URL}/generate_stream",
                f"{LLM_SPACE_URL}/api/generate",
                f"{LLM_SPACE_URL}/v1/completions",
                f"{LLM_SPACE_URL}/"
            ]
            
            for endpoint in endpoints_to_try:
                try:
                    async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                        if response.status_code == 200:
                            async for chunk in response.aiter_text():
                                if chunk.strip():
                                    try:
                                        # Try to parse as JSON (common format)
                                        if chunk.startswith("data: "):
                                            chunk = chunk[6:]  # Remove "data: " prefix
                                        
                                        if chunk.strip() == "[DONE]":
                                            break
                                        
                                        data = json.loads(chunk)
                                        
                                        # Extract text from various possible formats
                                        text = ""
                                        if "token" in data:
                                            text = data["token"]["text"]
                                        elif "choices" in data and len(data["choices"]) > 0:
                                            text = data["choices"][0].get("text", "")
                                        elif "generated_text" in data:
                                            text = data["generated_text"]
                                        elif isinstance(data, str):
                                            text = data
                                        
                                        if text:
                                            # Convert to Ollama format
                                            ollama_response = {
                                                "response": text,
                                                "done": False
                                            }
                                            yield f"data: {json.dumps(ollama_response)}\n\n"
                                        
                                    except json.JSONDecodeError:
                                        # Handle plain text responses
                                        if chunk.strip():
                                            ollama_response = {
                                                "response": chunk.strip(),
                                                "done": False
                                            }
                                            yield f"data: {json.dumps(ollama_response)}\n\n"
                            
                            # Send final "done" message
                            final_response = {"done": True}
                            yield f"data: {json.dumps(final_response)}\n\n"
                            return
                            
                except Exception as e:
                    print(f"Failed to connect to {endpoint}: {e}")
                    continue
            
            # If all endpoints failed, provide a fallback response
            fallback_response = {
                "response": f"I'm a proxy to {LLM_SPACE_URL}. Please ensure the LLM space is running and accessible.",
                "done": False
            }
            yield f"data: {json.dumps(fallback_response)}\n\n"
            
            final_response = {"done": True}
            yield f"data: {json.dumps(final_response)}\n\n"
            
    except Exception as e:
        error_response = {
            "response": f"Error connecting to LLM space: {str(e)}",
            "done": False
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        
        final_response = {"done": True}
        yield f"data: {json.dumps(final_response)}\n\n"

async def call_llm_space_once(prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
    """Call the LLM Space once and return a full, aggregated response (non-streaming)."""
    headers = {
        "Content-Type": "application/json",
    }
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "do_sample": True,
            "stream": False
        }
    }

    endpoints_to_try = [
        f"{LLM_SPACE_URL}/api/generate",
        f"{LLM_SPACE_URL}/v1/completions",
        f"{LLM_SPACE_URL}/",
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for endpoint in endpoints_to_try:
                try:
                    response = await client.post(endpoint, json=payload, headers=headers)
                    if response.status_code != 200:
                        continue

                    # Try to parse JSON formats commonly returned by HF Spaces
                    text = ""
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            if "choices" in data and data["choices"]:
                                # OpenAI-like format
                                choice = data["choices"][0]
                                text = choice.get("text") or choice.get("message", {}).get("content", "")
                            elif "generated_text" in data:
                                text = data["generated_text"]
                            elif "token" in data:
                                text = data["token"].get("text", "")
                            elif "outputs" in data and data["outputs"]:
                                text = data["outputs"][0].get("text", "")
                        elif isinstance(data, list) and data:
                            # Some Spaces return a list of generated items
                            item0 = data[0]
                            if isinstance(item0, dict):
                                text = item0.get("generated_text", "") or item0.get("text", "")
                    except Exception:
                        # Fallback to raw text
                        text = response.text

                    if not text:
                        # Attempt to concatenate possible newline-delimited JSON chunks
                        body = response.text
                        assembled = []
                        for line in body.splitlines():
                            line = line.strip()
                            if line.startswith("data: "):
                                line = line[6:]
                            if not line:
                                continue
                            try:
                                piece = json.loads(line)
                                if isinstance(piece, dict):
                                    piece_text = piece.get("text") or piece.get("generated_text")
                                    if not piece_text and "choices" in piece and piece["choices"]:
                                        piece_text = piece["choices"][0].get("text", "")
                                    if piece_text:
                                        assembled.append(piece_text)
                                elif isinstance(piece, str):
                                    assembled.append(piece)
                            except json.JSONDecodeError:
                                assembled.append(line)
                        text = "".join(assembled)

                    if text:
                        return text
                except Exception:
                    continue
    except Exception:
        pass

    return ""

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Generate endpoint matching Ollama's /api/generate"""
    if request.stream:
        return StreamingResponse(
            stream_from_llm_space(
                request.prompt, 
                request.temperature or 0.7,
                request.max_tokens or 150
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        # Non-streaming response: call the LLM once and return the full text
        text = await call_llm_space_once(
            request.prompt,
            request.temperature or 0.7,
            request.max_tokens or 150,
        )
        return {
            "response": text or "",
            "done": True,
        }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint matching Ollama's /api/chat"""
    # Convert messages to prompt
    prompt = messages_to_prompt(request.messages)
    
    if request.stream:
        return StreamingResponse(
            stream_from_llm_space(
                prompt,
                request.temperature or 0.7,
                request.max_tokens or 150
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        text = await call_llm_space_once(
            prompt,
            request.temperature or 0.7,
            request.max_tokens or 150,
        )
        return {
            "message": {"role": "assistant", "content": text or ""},
            "done": True,
        }

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "Ollama-compatible API Proxy running on Hugging Face Space",
        "llm_space_url": LLM_SPACE_URL,
        "endpoints": ["/api/generate", "/api/chat"],
        "note": "This proxy forwards requests to another HF Space running an LLM"
    }

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/tags")
async def list_models():
    """List available models (Ollama compatibility)"""
    return {
        "models": [
            {
                "name": "proxied-model",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 0,
                "details": {
                    "format": "proxy",
                    "family": "llm-space-proxy",
                    "parameter_size": "unknown"
                }
            }
        ]
    }

@app.get("/api/version")
async def version():
    """Return a version string compatible with Ollama clients."""
    return {"version": "0.1.0"}

@app.post("/api/pull")
async def pull_model(payload: dict):
    """No-op pull endpoint for clients that try to ensure a model is present."""
    name = payload.get("name", "proxied-model")
    return {
        "status": "success",
        "name": name,
        "digest": "",
    }

@app.post("/api/show")
async def show_model(payload: dict):
    """Return minimal model info to satisfy Ollama clients."""
    name = payload.get("name", "proxied-model")
    return {
        "model": name,
        "modified_at": "2024-01-01T00:00:00Z",
        "size": 0,
        "digest": "",
        "details": {
            "format": "proxy",
            "family": "llm-space-proxy",
        },
        "parameters": {},
        "template": "",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)