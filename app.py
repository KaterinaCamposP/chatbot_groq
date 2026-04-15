import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# 1. Configuración de entorno
load_dotenv()

# 2. Inicialización del LLM (Llama 3.3 via Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# 3. Configuración de Embeddings locales
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Procesamiento de la base de conocimientos
with open("panaderia_ositos.txt", "r", encoding="utf-8") as f:
    documento = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_text(documento)
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5. Definición del Prompt con Historial y Datos Corregidos
prompt = ChatPromptTemplate.from_template("""Eres el asistente virtual de Panadería Ositos. 
Tu trabajo es responder preguntas de los clientes ÚNICAMENTE usando la información proporcionada en el contexto y considerando el historial de la conversación.

Reglas estrictas:
1. SOLO responde con información que esté en el contexto.
2. Si la pregunta no se puede responder con el contexto, di:
   "Lo siento, no tengo esa información. Te recomiendo contactarnos por WhatsApp al +56 9 123 456 o por email a contacto@panaderiaositos.cl"
3. Sé amable, conciso y útil.
4. Si preguntan precios, menciona el precio exacto del contexto.
5. Responde en español.

Historial de conversación:
{history}

Contexto:
{context}

Pregunta: {question}
Respuesta:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Función para estructurar el historial de Gradio para el prompt
def format_history(history):
    formatted_history = ""
    for user_msg, bot_msg in history:
        formatted_history += f"Usuario: {user_msg}\nAsistente: {bot_msg}\n"
    return formatted_history

# 6. Función de respuesta con integración de Memoria
def respond(message, history):
    # Recuperamos fragmentos del archivo de texto
    docs = retriever.invoke(message)
    context = format_docs(docs)
    
    # Procesamos el historial acumulado
    history_str = format_history(history)
    
    # Ejecutamos la cadena de IA
    chain = prompt | llm | StrOutputParser()
    
    # Pasamos todas las variables al prompt
    return chain.invoke({
        "context": context,
        "history": history_str,
        "question": message
    })

# 7. Interfaz de Usuario
demo = gr.ChatInterface(
    fn=respond,
title="Panadería Ositos - Asistente Virtual 🐻",
    description="Pregúntame sobre nuestro menú, horarios y ubicación.",
    examples=[
        "¿Cuál es el horario de atención?",
        "¿Cuánto cuesta la Torta Osito?",
        "¿Dónde están ubicados?",
        "¿Tienen WhatsApp?",
        "¿Cuál es el precio de las dobladitas?",
        "¿Cuáles son los medios de pago?",
        "¿Hacen pedidos especiales?",
        "¿Qué sabores de mermelada tienen?"
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    
    # IMPORTANTE: server_name debe ser "0.0.0.0" para que Render pueda entrar
    demo.launch(server_name="0.0.0.0", server_port=port)