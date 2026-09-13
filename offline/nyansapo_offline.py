import pickle
import numpy as np
import torch
import torch.nn as nn
import requests
import time
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Load QIC-WCE classifier ────────────────────
# Point these to wherever you saved them from Drive

VECTORIZER_PATH = "/Users/michantech/downloads/qic_vectorizer.pkl"
MODEL_PATH      = "/Users/michantech/downloads/qic_wce_model.pt"
CHUNKS_PATH     = "/Users/michantech/downloads/chunks_v2.pkl"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "phi4-mini"

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x): return self.net(x)

print("Loading classifier...")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
qic_model = IntentClassifier(500, 256, 5)
qic_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
qic_model.eval()
print("✅ QIC-WCE classifier loaded")

print("Loading chunks...")
with open(CHUNKS_PATH, 'rb') as f:
    all_chunks = pickle.load(f)
print(f"✅ {len(all_chunks)} chunks loaded")

class_names = ['Definition','Explanation','Identification',
               'Procedural','Out-of-scope']

def classify_intent(question):
    vec  = vectorizer.transform([question]).toarray().astype(np.float32)
    x    = torch.tensor(vec)
    with torch.no_grad():
        logits = qic_model(x)
        pred   = logits.argmax(dim=1).item()
        conf   = torch.softmax(logits, dim=1).max().item()
    return class_names[pred], pred, round(conf, 3)

def bm25_retrieve(question, k=5):
    from rank_bm25 import BM25Okapi
    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    top    = np.argsort(scores)[::-1][:k]
    return [all_chunks[i] for i in top]

def generate_offline(question, chunks):
    context = "\n\n".join(chunks)
    prompt  = (
        f"You are a university tutor. Using ONLY the context below, "
        f"write one clear sentence answering the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 80}
        }, timeout=30)
        answer = r.json().get("response", "").strip()
        if not answer:
            return "I cannot find this in the lecture materials."
        return answer.split('\n')[0].strip()
    except Exception as e:
        return f"Offline generation error: {str(e)}"

# ── FastAPI ───────────────────────────────────────
app = FastAPI()

# Allow web app to call this API
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    t0 = time.time()
    intent, intent_class, confidence = classify_intent(q.question)
    if intent_class == 4:
        return {
            "question": q.question,
            "intent": intent,
            "intent_confidence": confidence,
            "answer": "I cannot find this in the lecture materials.",
            "source": "Blocked by Nyansapo-QIC-WCE — out-of-scope query",
            "crag_triggered": False,
            "latency": round(time.time() - t0, 2),
            "mode": "offline"
        }
    chunks = bm25_retrieve(q.question, k=5)
    answer = generate_offline(q.question, chunks)
    return {
        "question": q.question,
        "intent": intent,
        "intent_confidence": confidence,
        "answer": answer,
        "source": f"Generated from {len(chunks)} lecture chunks (offline)",
        "crag_triggered": False,
        "latency": round(time.time() - t0, 2),
        "mode": "offline"
    }

@app.get("/health")
def health():
    return {
        "status": "live",
        "mode": "offline",
        "classifier": "Nyansapo-QIC-WCE",
        "generator": f"Phi-4-mini via Ollama",
        "chunks": len(all_chunks)
    }

if __name__ == "__main__":
    print("\n✅ Nyansapo offline API starting...")
    print("   Ask endpoint : http://localhost:8000/ask")
    print("   Health check : http://localhost:8000/health")
    print("\n   Paste this into the web app: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)