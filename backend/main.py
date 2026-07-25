from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, asyncio, json, requests, threading
from datetime import datetime
import sqlite_utils

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
OLLAMA_URL = "http://localhost:11434"

def get_db():
    db = sqlite_utils.Database("/memory/agentoffice.db")
    if "tasks" not in db.table_names():
        db["tasks"].create({"id": int, "title": str, "status": str, "agent": str, "result": str, "created_at": str, "updated_at": str}, pk="id")
    if "agent_status" not in db.table_names():
        db["agent_status"].create({"name": str, "status": str, "current_task": str, "updated_at": str}, pk="name")
    return db

def run_agent(task_id: int, title: str):
    db = get_db()
    db["agent_status"].upsert({"name": "Coder", "status": "working", "current_task": title, "updated_at": datetime.now().isoformat()}, pk="name")
    db["tasks"].update(task_id, {"status": "running", "agent": "Coder", "updated_at": datetime.now().isoformat()})
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": "qwen2.5-coder:7b", "prompt": title, "stream": False}, timeout=300)
        result = resp.json().get("response", "Keine Antwort")
    except Exception as e:
        result = f"Fehler: {str(e)}"
    db["tasks"].update(task_id, {"status": "done", "result": result, "updated_at": datetime.now().isoformat()})
    db["agent_status"].upsert({"name": "Coder", "status": "sleeping", "current_task": "", "updated_at": datetime.now().isoformat()}, pk="name")

@app.get("/status")
def get_status():
    db = get_db()
    return {"agents": list(db["agent_status"].rows), "active_tasks": list(db["tasks"].rows_where("status = 'running'"))}

@app.get("/tasks")
def get_tasks():
    return list(get_db()["tasks"].rows)

@app.post("/task")
def add_task(task: dict):
    db = get_db()
    row_id = db["tasks"].insert({"title": task["title"], "status": "pending", "agent": "", "result": "", "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()}).last_pk
    threading.Thread(target=run_agent, args=(row_id, task["title"])).start()
    return {"ok": True, "id": row_id}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        db = get_db()
        await websocket.send_text(json.dumps({"agents": list(db["agent_status"].rows)}))
        await asyncio.sleep(2)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
