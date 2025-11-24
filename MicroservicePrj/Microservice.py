from fastapi import FastAPI,Query,Request
from fastapi.middleware.cors import CORSMiddleware
from Anonimyzer import Anonymizer
from Request_response_Entity import * 
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.anonymizer = Anonymizer(use_gpu=True, device=0)
    try:
        yield
    finally: 
        anonymizer = app.state.anonymizer
        del anonymizer

app = FastAPI(
    title="LLMed-anon",
    description="Microservizio per effettuare anonimizzazione sui referti",
    version="0.0.1",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette richieste da qualsiasi dominio (in sviluppo)
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi (GET, POST, ecc.)
    allow_headers=["*"],  # Permette tutti gli headers
)

@app.post("/anonimize",description="End-point per effettuare una anonimizzazione di una frase inviata")
async def anonimize(payload: AnonimizeRequest):
    anonymizer: Anonymizer = app.state.anonymizer
    text = payload.text
    resp = AnonimizeResponse(anonymized_text = anonymizer.anonymize_text(text))
    return resp.model_dump_json()