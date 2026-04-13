# 🐻 Panadería Ositos - AI Assistant (RAG)

An intelligent virtual assistant designed to optimize customer service for **Panadería Ositos**. This project implements a **RAG (Retrieval-Augmented Generation)** architecture to provide precise answers regarding products, pricing, and business hours, effectively eliminating LLM hallucinations by grounding responses in a verified knowledge base.

## 🚀 Tech Stack
* **LLM:** Llama 3.3 (70B) via [Groq Cloud](https://groq.com/) for ultra-low latency inference.
* **Embeddings:** `all-MiniLM-L6-v2` (Hugging Face) processed locally for semantic search.
* **Vector Store:** FAISS (Facebook AI Similarity Search) for efficient document retrieval.
* **Orchestration:** LangChain for managing the RAG chain and prompt templates.
* **Interface:** Gradio for a streamlined, web-based chat UI.

## 🛠️ Technical Features
- **Conversational Memory:** Implemented stateful logic to handle follow-up questions and maintain context throughout the chat session.
- **Context Injection:** Automated parsing and indexing of unstructured data from a localized knowledge source (`.txt`).
- **Strict Logic Prompting:** Engineered system prompts to ensure responses are strictly derived from the provided context.
- **Environment Management:** Secure handling of API credentials using `.env` and `python-dotenv`.
