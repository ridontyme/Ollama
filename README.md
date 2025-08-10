---
title: Ollama-Compatible API Proxy
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# Ollama-Compatible API Proxy for Hugging Face Spaces

This FastAPI application acts as a proxy that provides an Ollama-compatible HTTP API while forwarding requests to another Hugging Face Space running an LLM. This allows mobile apps expecting an Ollama server to connect to any HF Space LLM.

## 🏗 Architecture
1. **Space A** (This code): Ollama-compatible API proxy
2. **Space B** (Your LLM): Any Hugging Face Space running a language model
3. **iOS App**: Uses Space A's URL as the Ollama server address

## 🚀 Setup Instructions

### Step 1: Deploy Your LLM Space (Space B)
First, create or find a Hugging Face Space running your desired LLM. Popular options:
- Mistral-7B spaces
- Llama-2 spaces  
- CodeLlama spaces
- Any custom model space

Note the Space URL: `https://username-llmspace.hf.space`

**For this configuration, we're using:**
- **Space B (LLM)**: AMD GPT-OSS 120B Chatbot (`https://amd-gpt-oss-120b-chatbot.hf.space`)
- This is a powerful 120B parameter model that will provide high-quality responses

### Step 2: Deploy This Proxy Space (Space A)
1. Create a new Hugging Face Space
2. Upload these files:
   - `main.py`
   - `requirements.txt` 
   - `README.md`
3. Set environment variables in your Space settings:
   - `LLM_SPACE_URL`: URL of your LLM space (default: `https://amd-gpt-oss-120b-chatbot.hf.space`)
   - `HF_TOKEN`: (Optional) Your Hugging Face token if the LLM space is private

### Step 3: Connect Your iOS App
In your iOS app, use this Space's URL as the Ollama server:

```swift
// Replace with your proxy space URL
let ollamaURL = "https://username-proxyspace.hf.space"

// Use exactly like a regular Ollama server
let request = ChatRequest(
    model: "any-model-name",
    messages: [
        ChatMessage(role: "user", content: "Hello!")
    ],
    stream: true
)
📡 API Endpoints
POST /api/generate
Generate text from a prompt (matches Ollama’s /api/generate endpoint).
Request:
{
  "model": "any-model-name",
  "prompt": "Hello, how are you?",
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 150
}
POST /api/chat
Chat with the model using conversation format (matches Ollama’s /api/chat endpoint).
Request:
{
  "model": "any-model-name", 
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 150
}
Response Format: Both endpoints return Server-Sent Events exactly like Ollama:
data: {"response": "Hello", "done": false}
data: {"response": " there!", "done": false}
data: {"done": true}
GET /api/tags
List available models (Ollama compatibility).
🔧 Configuration
Set these environment variables in your Hugging Face Space:
•  LLM_SPACE_URL (Required): The URL of your LLM space
https://amd-gpt-oss-120b-chatbot.hf.space
•  HF_TOKEN (Optional): Your Hugging Face token for private spaces
hf_xxxxxxxxxxxxxxxxxxxx
🎯 Compatible LLM Spaces
This proxy works with LLM spaces that provide:
•  HTTP POST endpoints for text generation
•  Streaming responses (preferred)
•  JSON or plain text output
Common endpoint patterns it tries:
•  /generate_stream
•  /api/generate
•  /v1/completions
•  / (root endpoint)
📱 iOS App Integration
Your iOS app can now use this proxy exactly like a local Ollama instance:
// Instead of: http://localhost:11434
// Use your proxy space URL
let baseURL = "https://your-proxy-space.hf.space"

// All Ollama client libraries work unchanged
let client = OllamaClient(baseURL: baseURL)
🔄 Flow Example
1.  iOS app sends chat request to https://your-proxy-space.hf.space/api/chat
2.  Proxy converts request and forwards to https://amd-gpt-oss-120b-chatbot.hf.space
3.  AMD GPT-OSS 120B model generates response and streams back
4.  Proxy converts response to Ollama format and streams to iOS app
5.  iOS app receives standard Ollama-formatted streaming response
⚡ Benefits
•  Zero iOS app changes: Works with existing Ollama client code
•  Model flexibility: Switch LLM spaces without app updates
•  Scalability: Use powerful HF Space GPUs instead of local compute
•  Cost effective: Pay only for HF Space usage, no local GPU needed
•  Easy deployment: No complex server setup required
🛠 Troubleshooting
Connection Issues:
•  Verify LLM_SPACE_URL is correct and accessible
•  Check if LLM space requires authentication (set HF_TOKEN)
•  Ensure LLM space is running and not sleeping
Response Issues:
•  Check LLM space logs for errors
•  Verify the LLM space accepts the request format
•  Try different endpoint patterns in the proxy code
iOS App Issues:
•  Ensure CORS is working (should be enabled by default)
•  Check network connectivity to HF Spaces
•  Verify the app is using HTTPS (required for HF Spaces)
🔒 Security Notes
•  This proxy enables CORS for all origins (required for mobile apps)
•  Use HTTPS only (HF Spaces provide this automatically)
•  Keep HF tokens secure and use environment variables
•  Consider rate limiting for production use
### Key Additions and Notes
- **YAML Block**: Added at the top of the file, enclosed in `---` delimiters, with the following parameters:
  - `title`: Descriptive name for the Space.
  - `emoji`: Rocket emoji for thumbnail appeal.
  - `colorFrom` and `colorTo`: Blue-to-purple gradient for the thumbnail.
  - `sdk`: Set to `docker` since your FastAPI app runs in a custom Docker environment.
  - `app_port`: Set to `7860`, the default port for Hugging Face Spaces (confirm this matches your `main.py` or Dockerfile).
  - `custom_headers`: Added CORS-related headers to support your security notes and ensure compatibility with iOS app integration.
- **Preserved Content**: The rest of your README remains unchanged, maintaining all setup instructions, API endpoint details, and troubleshooting guidance.
- **Next Steps**:
  1. Replace the existing `README.md` in your Hugging Face Space repository with this updated version.
  2. Ensure `main.py` and `requirements.txt` are correctly uploaded to the repository root.
  3. Set the environment variables (`LLM_SPACE_URL` and optionally `HF_TOKEN`) in the Space's settings under the "Variables" section.
  4. Restart the Space and check the build logs for any remaining errors.
- **CORS in FastAPI**: As a precaution, verify that your `main.py` includes CORS middleware to allow requests from all origins, as mentioned in your security notes. Example:
  ```python
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware

  app = FastAPI()

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
