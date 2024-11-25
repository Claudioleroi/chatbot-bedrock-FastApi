from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import os
import shutil
import queue
import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional
import psutil
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import boto3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ServiceClientAI")

app = FastAPI(title="Service Client AI Assistant API")

# Shared resources
file_contents: Dict[str, pd.DataFrame] = {}
company_name: Optional[str] = None
log_queue = queue.Queue()
thread_pool = asyncio.get_event_loop()

# AWS Bedrock settings
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "us.meta.llama3-2-3b-instruct-v1:0")

# Bedrock client setup
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging handler to send logs to the queue
class CustomHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "process_id": os.getpid(),
            "thread_id": record.thread,
            "memory_usage": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
        }
        log_queue.put(log_entry)

logger.addHandler(CustomHandler())

# Context management for query and file data
class ContextManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.document_vectors = {}

    def update_vectors(self, file_name: str, content: str):
        if not self.document_vectors:
            self.vectorizer.fit([content])
        vector = self.vectorizer.transform([content])
        self.document_vectors[file_name] = vector

    def get_relevant_context(self, query: str, file_names: Optional[List[str]] = None, max_length: int = 2000) -> str:
        if not self.document_vectors:
            return ""
        query_vector = self.vectorizer.transform([query])
        relevant_parts = []

        files_to_search = file_names if file_names else self.document_vectors.keys()
        for file_name in files_to_search:
            if file_name in self.document_vectors:
                similarity = cosine_similarity(query_vector, self.document_vectors[file_name])[0][0]
                if similarity > 0.1:  # Threshold for relevance
                    content = file_contents[file_name]["text"].str.cat(sep=" ") if "text" in file_contents[file_name].columns else file_contents[file_name].to_string()
                    relevant_parts.append(content[:max_length])

        return "\n\n".join(relevant_parts)

context_manager = ContextManager()

def format_prompt_with_context(text: str, language: str, context: str) -> str:
    assistant_intro = f"Je suis l'assistant de {company_name}. " if company_name else "Je suis votre assistant."
    if language == "fr":
        return f"{assistant_intro}\n\n{text}\n\nMerci de répondre de manière concise et professionnelle en quelques phrases."
    return f"{assistant_intro}\n\n{text}\n\nAnswer concisely and professionally in a few sentences."
async def generate_bedrock_response(prompt: str) -> str:
    try:
        payload = {"prompt": prompt}
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        
        # Nettoyer et réduire la réponse
        response_text = response_body.get("generation", "")
        response_text = re.sub(r'^(Réponse:|Response:)\s*', '', response_text, flags=re.IGNORECASE)
        
        # Limiter à la première phrase ou aux deux premières phrases
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        concise_response = ' '.join(sentences[:2]).strip()
        
        return concise_response if concise_response else "Je suis désolé, je ne peux pas répondre à cette question."
    except Exception as e:
        logger.error(f"Error in Bedrock inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")
user_language = {}  # Stores language preference per user session

class Query(BaseModel):
    text: str
    max_length: int = 10000
    language: str = "fr"
    file_names: Optional[List[str]] = None

@app.post("/predict")
async def predict(request: Request, query: Query):
    try:
        logger.info(f"Received prediction request: {query.text[:50]}...")
        user_id = request.client.host
        language = user_language.get(user_id, query.language)
        context = context_manager.get_relevant_context(query.text, query.file_names)
        formatted_prompt = format_prompt_with_context(query.text, language, context)
        response_text = await generate_bedrock_response(formatted_prompt)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-training-data")
async def upload_training_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global company_name
    try:
        logger.info(f"Receiving file upload: {file.filename}")
        os.makedirs("temp_data", exist_ok=True)
        file_path = f"temp_data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        data = await process_file_content(file_path, file.filename)
        file_contents[file.filename] = data
        text_content = data["text"].str.cat(sep=" ") if "text" in data.columns else data.to_string()
        background_tasks.add_task(context_manager.update_vectors, file.filename, text_content)
        if "text" in data.columns:
            company_name = extract_company_name(text_content) or company_name
        return {"message": "File uploaded and processed successfully", "filename": file.filename, "company_name": company_name}
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_file_content(file_path: str, file_name: str) -> pd.DataFrame:
    if file_name.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_name.endswith(".txt"):
        with open(file_path, "r") as f:
            text_content = f.read()
        return pd.DataFrame({"text": text_content.splitlines()})
    elif file_name.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        return pd.DataFrame({"text": text.splitlines()})
    else:
        raise ValueError("Unsupported file format")

def extract_company_name(text: str) -> Optional[str]:
    match = re.search(r"(Company Name|Organization|Client Name):\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    first_line = text.splitlines()[0].strip()
    if first_line:
        return first_line

@app.get("/logs")
async def stream_logs():
    async def log_stream() -> AsyncGenerator[str, None]:
        while True:
            try:
                if not log_queue.empty():
                    log_entry = log_queue.get_nowait()
                    yield f"data: {json.dumps(log_entry)}\n\n"
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in log stream: {e}")
                yield f"data: {json.dumps({'level': 'ERROR', 'message': str(e)})}\n\n"

    return StreamingResponse(log_stream(), media_type="text/event-stream")

# Run FastAPI with uvicorn
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)