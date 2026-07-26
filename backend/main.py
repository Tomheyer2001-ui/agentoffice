from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn, asyncio, json, requests, threading, os, shutil, time
import sys, subprocess, tempfile, re, math
from datetime import datetime
from pathlib import Path
import sqlite_utils

try:
    import psutil
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

try:
    import docker as dockerlib
    docker_client = dockerlib.from_env()
except:
    docker_client = None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"])

OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_CODE    = "qwen2.5-coder:7b"
MODEL_VISION  = "llava:7b"
MODEL_EMBED   = "nomic-embed-text"
SEARXNG_URL   = "http://localhost:8888"
N8N_URL       = os.getenv("N8N_URL", "http://localhost:5678")
N8N_API_KEY   = os.getenv("N8N_API_KEY", "")
DB_PATH       = "/memory/agentoffice.db"
FRONTEND_PATH = "/memory/index.html"
FRONTEND_SRC  = "/app/frontend/index.html"
INBOX         = Path("/inbox")
OUTBOX        = Path("/outbox")
WORKSPACE     = Path("/workspace")
IMPROVE_AFTER = 5
CONFIDENCE_THRESHOLD = 6
DEBATE_THRESHOLD     = 8
MAX_TOOL_STEPS       = 6
RESOURCE_WARN_RAM    = 85

NO_QUESTIONS = (
    "\n\nWICHTIG: Stelle NIEMALS Rückfragen. "
    "Triff alle Entscheidungen selbst. Arbeite vollständig durch.")

SLEEP_MODE      = False
TASK_QUEUE      = []
TASK_QUEUE_LOCK = threading.Lock()
PAUSED_TASKS    = set()
RESOURCE_STATUS = {"ram_pct": 0, "cpu_pct": 0, "warning": False}

FILE_EXTENSIONS = {
    "video":[".mp4",".mov",".avi",".mkv",".webm"],
    "audio":[".mp3",".wav",".flac",".m4a",".ogg"],
    "image":[".jpg",".jpeg",".png",".gif",".webp",".svg"],
    "code": [".py",".js",".html",".css",".json",".sh",".ts"],
    "text": [".txt",".md",".csv",".xml"],
    "doc":  [".pdf",".docx",".xlsx"],
}

DEFAULT_TEMPLATES = [
    {"name":"Video schneiden","description":"Video mit FFmpeg schneiden","category":"video",
     "steps":["Video analysieren","Schnittplan erstellen","FFmpeg ausführen","Prüfen"]},
    {"name":"Python Tool","description":"Python-Script erstellen","category":"code",
     "steps":["Anforderungen analysieren","Implementieren","Tests schreiben","Dokumentieren"]},
    {"name":"Spiel entwickeln","description":"Spiel in Python/HTML5","category":"game",
     "steps":["Konzept definieren","Grundstruktur","Grafik und UI","Testen"]},
    {"name":"Audio transkribieren","description":"Whisper-Transkription","category":"audio",
     "steps":["Transkribieren","Strukturieren","Keywords","Exportieren"]},
    {"name":"Web Scraper","description":"Daten von Webseite sammeln","category":"code",
     "steps":["Seite analysieren","Scraper schreiben","Bereinigen","Exportieren"]},
    {"name":"n8n Workflow","description":"n8n-Workflow via API erstellen","category":"automation",
     "steps":["Anforderungen","JSON generieren","Via API erstellen","Testen"]},
]

QUICK_TASKS = [
    {"label":"Code prüfen","icon":"⌨","prompt":"Überprüfe den Code in der Inbox auf Fehler"},
    {"label":"Video kürzen","icon":"▶","prompt":"Schneide das Video in der Inbox"},
    {"label":"Transkribieren","icon":"♪","prompt":"Transkribiere die Audiodatei mit Whisper"},
    {"label":"Bild analysieren","icon":"◻","prompt":"Analysiere das Bild mit LLaVA detailliert"},
    {"label":"n8n Workflow","icon":"⚙","prompt":"Erstelle einen n8n-Workflow für Dateiverarbeitung"},
    {"label":"Status Bericht","icon":"≡","prompt":"Erstelle Zusammenfassung aller Arbeiten und Empfehlungen"},
]

AGENT_TOOLS = [
    {"name":"search_web","description":"Sucht im Web","parameters":{"query":"Suchanfrage"}},
    {"name":"run_python","description":"Führt Python-Code aus","parameters":{"code":"Python-Code"}},
    {"name":"read_file","description":"Liest Datei aus Workspace","parameters":{"filename":"Dateiname"}},
    {"name":"write_file","description":"Schreibt Datei in Workspace","parameters":{"filename":"Name","content":"Inhalt"}},
    {"name":"call_api","description":"HTTP-Request","parameters":{"url":"URL","method":"GET/POST","data":"JSON"}},
    {"name":"ffmpeg_run","description":"Führt FFmpeg-Befehl aus","parameters":{"command":"ffmpeg ..."}},
]

def file_type(path: Path) -> str:
    ext = path.suffix.lower()
    for ftype, exts in FILE_EXTENSIONS.items():
        if ext in exts: return ftype
    return "other"

# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_db():
    db = sqlite_utils.Database(DB_PATH)
    tables = db.table_names()

    if "projects" not in tables:
        db["projects"].create({"id":int,"name":str,"description":str,
            "status":str,"created_at":str,"updated_at":str},pk="id")
    if "project_steps" not in tables:
        db["project_steps"].create({"id":int,"project_id":int,"step_num":int,
            "title":str,"status":str,"result":str,"updated_at":str},pk="id")
    if "tasks" not in tables:
        db["tasks"].create({"id":int,"project_id":int,"title":str,"status":str,
            "priority":int,"paused":int,"agent":str,"result":str,"research":str,
            "skills_used":str,"feedback":str,"source_file":str,"output_file":str,
            "model_used":str,"exec_output":str,"confidence":int,"depends_on":str,
            "created_at":str,"updated_at":str},pk="id")
    if "task_checkpoints" not in tables:
        db["task_checkpoints"].create({"id":int,"task_id":int,"step":str,
            "data":str,"created_at":str},pk="id")
    if "agent_status" not in tables:
        db["agent_status"].create({"name":str,"status":str,
            "current_task":str,"skills":str,"updated_at":str},pk="name")
    else:
        # Migration: fehlende Spalten hinzufügen
        existing_cols = [c.name for c in db["agent_status"].columns]
        if "skills" not in existing_cols:
            db["agent_status"].add_column("skills", str, not_null=False)
        if "current_task" not in existing_cols:
            db["agent_status"].add_column("current_task", str, not_null=False)

    if "agent_roster" not in tables:
        db["agent_roster"].create({"name":str,"role":str,"specialization":str,
            "status":str,"system_prompt":str,"hired_for_task":int,
            "tasks_completed":int,"avg_quality":float,
            "hired_at":str,"retired_at":str},pk="name")
        for a in [
            {"name":"Chef","role":"chef",
             "specialization":"Orchestrierung & Qualitätskontrolle",
             "system_prompt":"Du bist der Chef-Agent. Du koordinierst das Team, bewertest Qualität und gibst finale Genehmigungen."},
            {"name":"Coder","role":"coder",
             "specialization":"Python, Code, Skripte",
             "system_prompt":"Du bist ein präziser Code-Agent. Du löst Programmieraufgaben vollständig und ohne Rückfragen."},
            {"name":"Critic","role":"critic",
             "specialization":"Code-Review & Fehleranalyse",
             "system_prompt":"Du bist ein kritischer Reviewer. Du findest Fehler, Lücken und Qualitätsprobleme."},
            {"name":"Judge","role":"judge",
             "specialization":"Skills & Selbstverbesserung",
             "system_prompt":"Du bist der Judge-Agent. Du schreibst und verbesserst Skills und System-Prompts."},
            {"name":"Planner","role":"planner",
             "specialization":"Recherche & Projektplanung",
             "system_prompt":"Du bist ein Recherche-Experte. Du findest die besten Ansätze und planst Projekte."},
        ]:
            db["agent_roster"].insert({**a,"status":"active",
                "hired_for_task":None,"tasks_completed":0,"avg_quality":0.0,
                "hired_at":datetime.now().isoformat(),"retired_at":""})
    if "agent_messages" not in tables:
        db["agent_messages"].create({"id":int,"from_agent":str,"to_agent":str,
            "task_id":int,"message_type":str,"content":str,"status":str,
            "created_at":str},pk="id")
    if "skills" not in tables:
        db["skills"].create({"id":int,"name":str,"category":str,"description":str,
            "how_to":str,"tools_needed":str,"pitfalls":str,"examples":str,
            "embedding":str,"success_count":int,"failure_count":int,"version":int,
            "last_used":str,"created_at":str},pk="id")
    if "model_performance" not in tables:
        db["model_performance"].create({"id":int,"model":str,"task_type":str,
            "quality":int,"success":int,"duration_s":float,
            "recorded_at":str},pk="id")
    if "improvements" not in tables:
        db["improvements"].create({"id":int,"summary":str,"applied_at":str},pk="id")
    if "tools" not in tables:
        db["tools"].create({"name":str,"description":str,"api_url":str,
            "container_name":str,"stack_name":str,"status":str,"auto_stop":int,
            "container_pattern":str,"install_hint":str,"updated_at":str},pk="name")
        for t in [
            {"name":"chatterbox-tts","description":"TTS mit Voice Cloning",
             "api_url":"http://localhost:8055","auto_stop":True,
             "container_pattern":"chatterbox","stack_name":"chatterbox-tts","install_hint":"Umbrel-App"},
            {"name":"whisper-asr","description":"Spracherkennung",
             "api_url":"http://localhost:9000","auto_stop":True,
             "container_pattern":"whisper","stack_name":"whisper","install_hint":"Umbrel-App"},
            {"name":"n8n","description":"Workflow-Automation",
             "api_url":"http://localhost:5678","auto_stop":False,
             "container_pattern":"n8n_server","stack_name":"n8n","install_hint":"Umbrel-App"},
        ]:
            db["tools"].upsert({"name":t["name"],"description":t["description"],
                "api_url":t["api_url"],"container_name":"","stack_name":t["stack_name"],
                "status":"unknown","auto_stop":int(t["auto_stop"]),
                "container_pattern":t["container_pattern"],
                "install_hint":t["install_hint"],"updated_at":datetime.now().isoformat()},pk="name")
    if "tool_requests" not in tables:
        db["tool_requests"].create({"id":int,"tool_name":str,"reason":str,
            "requested_by":str,"status":str,"install_hint":str,"created_at":str},pk="id")
    if "code_updates" not in tables:
        db["code_updates"].create({"id":int,"title":str,"description":str,
            "new_code":str,"status":str,"proposed_by":str,
            "created_at":str,"reviewed_at":str},pk="id")
    if "inbox_log" not in tables:
        db["inbox_log"].create({"filename":str,"processed_at":str},pk="filename")
    if "sleep_schedules" not in tables:
        db["sleep_schedules"].create({"id":int,"name":str,"weekdays":str,
            "start_time":str,"end_time":str,"active":int,"created_at":str},pk="id")
    if "journal" not in tables:
        db["journal"].create({"id":int,"content":str,"type":str,
            "author":str,"tags":str,"created_at":str},pk="id")
    if "project_templates" not in tables:
        db["project_templates"].create({"id":int,"name":str,"description":str,
            "category":str,"steps":str,"created_at":str},pk="id")
        for t in DEFAULT_TEMPLATES:
            db["project_templates"].insert({"name":t["name"],"description":t["description"],
                "category":t["category"],"steps":json.dumps(t["steps"]),
                "created_at":datetime.now().isoformat()})
    if "settings" not in tables:
        db["settings"].create({"key":str,"value":str},pk="key")
        db["settings"].insert({"key":"thinking_mode","value":"auto"})
    return db

def set_agent(name, status, task=""):
    """Setzt Agent-Status – fängt alle DB-Fehler ab damit Threads nicht abstürzen."""
    try:
        db = get_db()
        db["agent_status"].upsert({
            "name": name, "status": status,
            "current_task": task, "skills": "",
            "updated_at": datetime.now().isoformat()}, pk="name")
    except Exception as e:
        print(f"set_agent error ({name}/{status}): {e}")

def get_agent_prompt(name):
    try: return get_db()["agent_roster"].get(name)["system_prompt"] or ""
    except: return ""

def save_checkpoint(task_id, step, data):
    try:
        get_db()["task_checkpoints"].insert({"task_id":task_id,"step":step,
            "data":json.dumps(data) if not isinstance(data,str) else data,
            "created_at":datetime.now().isoformat()})
    except Exception as e:
        print(f"save_checkpoint error: {e}")

def load_checkpoint(task_id, step):
    try:
        db = get_db()
        rows = list(db["task_checkpoints"].rows_where(
            "task_id=? AND step=?",[task_id,step],order_by="created_at desc"))
        if rows:
            try: return json.loads(rows[0]["data"])
            except: return rows[0]["data"]
    except: pass
    return None

# ── LLM ───────────────────────────────────────────────────────────────────────
def llm(prompt, system="Du bist ein hilfreicher KI-Assistent.", model=None):
    m = model or MODEL_CODE
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",json={
            "model":m,"prompt":prompt,
            "system":system+NO_QUESTIONS,"stream":False},timeout=1800)
        return r.json().get("response","")
    except Exception as e: return f"Fehler: {e}"

def llm_vision(prompt, image_path):
    import base64
    try:
        with open(image_path,"rb") as f: img=base64.b64encode(f.read()).decode()
        r = requests.post(f"{OLLAMA_URL}/api/generate",json={
            "model":MODEL_VISION,"prompt":prompt+NO_QUESTIONS,
            "images":[img],"stream":False},timeout=1800)
        return r.json().get("response","")
    except Exception as e: return f"Vision-Fehler: {e}"

def get_embedding(text: str) -> list:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
            json={"model":MODEL_EMBED,"prompt":text[:2000]},timeout=60)
        return r.json().get("embedding",[])
    except: return []

def cosine_similarity(a, b) -> float:
    if not a or not b or len(a)!=len(b): return 0.0
    dot  = sum(x*y for x,y in zip(a,b))
    na   = math.sqrt(sum(x*x for x in a))
    nb   = math.sqrt(sum(x*x for x in b))
    if na==0 or nb==0: return 0.0
    return dot/(na*nb)

def select_model(task_title, file_type_hint=None):
    title_lower = task_title.lower()
    vision_kw = ["bild","foto","image","screenshot","photo","visuell","grafik"]
    if file_type_hint=="image" or any(w in title_lower for w in vision_kw):
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags",timeout=3)
            models = [m["name"] for m in r.json().get("models",[])]
            if any("llava" in m for m in models): return MODEL_VISION
        except: pass
    return MODEL_CODE

# ── TOOL CALLING ──────────────────────────────────────────────────────────────
def execute_tool(name, args) -> str:
    try:
        if name=="search_web":
            return search_web(args.get("query","")) or "Keine Ergebnisse."
        elif name=="run_python":
            res = execute_code(args.get("code",""))
            return f"Output:\n{res['stdout'][:800]}" if res["success"] else f"Fehler: {res['stderr'][:400]}"
        elif name=="read_file":
            p = WORKSPACE/args.get("filename","")
            if p.exists(): return p.read_text(errors="ignore")[:2000]
            return "Datei nicht gefunden."
        elif name=="write_file":
            p = WORKSPACE/args.get("filename","output.txt")
            p.write_text(args.get("content",""))
            return f"Geschrieben: {p.name}"
        elif name=="call_api":
            m = args.get("method","GET").upper()
            res = requests.get(args["url"],timeout=10) if m=="GET" \
                else requests.post(args["url"],json=args.get("data",{}),timeout=10)
            return res.text[:1000]
        elif name=="ffmpeg_run":
            cmd=args.get("command","")
            if not cmd.startswith("ffmpeg"): return "Ungültiger Befehl."
            r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=300)
            return r.stdout[:500] if r.returncode==0 else f"Fehler: {r.stderr[:300]}"
    except Exception as e:
        return f"Tool-Fehler ({name}): {e}"
    return "Unbekanntes Tool."

def llm_with_tools(task, system, model=None, agent_name="Coder") -> tuple:
    m = model or MODEL_CODE
    tools_desc = "\n".join([
        f"- {t['name']}: {t['description']} | Parameter: {t['parameters']}"
        for t in AGENT_TOOLS])
    tool_log = []
    context_parts = []
    base = (
        f"Aufgabe: {task}\n\nVerfügbare Tools:\n{tools_desc}\n\n"
        "Wenn du ein Tool brauchst, antworte NUR mit JSON:\n"
        '{"tool":"name","args":{"key":"value"}}\n'
        "Wenn du fertig bist, antworte normal (kein JSON-Tool-Aufruf).")
    for step in range(MAX_TOOL_STEPS):
        prompt = base
        if context_parts:
            prompt += "\n\nBisherige Tool-Ergebnisse:\n" + "\n".join(context_parts)
        response = llm(prompt, system, m)
        m_json = re.search(r'\{\s*"tool"\s*:', response)
        if m_json:
            try:
                call = json.loads(response.strip().strip("```json").strip("```").strip())
                if "tool" in call:
                    result = execute_tool(call["tool"], call.get("args",{}))
                    log_entry = f"[{call['tool']}] → {result[:300]}"
                    tool_log.append(log_entry)
                    context_parts.append(log_entry)
                    continue
            except: pass
        return response, tool_log
    final = llm(f"Aufgabe: {task}\n\nErkenntnisse:\n"+"\n".join(context_parts)+
                "\n\nFasse die Lösung zusammen:", system, m)
    return final, tool_log

# ── CONFIDENCE ────────────────────────────────────────────────────────────────
def assess_confidence(task, result, model=None) -> int:
    try:
        raw = llm(
            f"Aufgabe: {task}\nLösung: {result[:500]}\n\n"
            "Wie sicher bist du dass diese Lösung korrekt ist? Bewerte 1-10. NUR eine Zahl.",
            "Selbsteinschätzung. Nur eine Zahl.", model or MODEL_CODE)
        return max(1,min(10,int(re.search(r'\d+',raw).group())))
    except: return 7

# ── RAG ───────────────────────────────────────────────────────────────────────
def update_skill_embedding(skill_id, text):
    try:
        emb = get_embedding(text)
        if emb:
            get_db()["skills"].update(skill_id,{"embedding":json.dumps(emb)})
    except: pass

def get_relevant_skills(title: str) -> str:
    try:
        db = get_db()
        skills = list(db["skills"].rows)
        if not skills: return ""
        query_emb = get_embedding(title)
        if query_emb:
            scored = []
            for s in skills:
                try:
                    semb = json.loads(s.get("embedding") or "[]")
                    score = cosine_similarity(query_emb, semb)
                    scored.append((score, s))
                except:
                    scored.append((0.0, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            relevant = [s for sc,s in scored if sc > 0.3][:4]
        else:
            overview = "\n".join([f"[{s['id']}] {s['name']}: {s['description']}" for s in skills])
            raw = llm(f"Aufgabe: {title}\nSkills:\n{overview}\nRelevante IDs? NUR JSON oder [].",
                      "Nur JSON.")
            try:
                ids = json.loads(raw.strip().strip("```json").strip("```").strip())
                relevant = [s for s in skills if s["id"] in ids][:4]
            except: relevant = []
        if not relevant: return ""
        return "\n\n--- Relevante Skills ---\n" + "\n\n".join([
            f"=== {s['name']} (v{s['version']}, {s['success_count']}x OK) ===\n"
            f"Wie: {s['how_to']}\nTools: {s['tools_needed']}\nFallstricke: {s['pitfalls']}"
            for s in relevant])
    except Exception as e:
        print(f"get_relevant_skills error: {e}")
        return ""

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
def record_model_performance(model, task_type, quality, success, duration_s):
    try:
        get_db()["model_performance"].insert({"model":model,"task_type":task_type,
            "quality":quality,"success":int(success),"duration_s":duration_s,
            "recorded_at":datetime.now().isoformat()})
    except: pass

def best_model_for_type(task_type: str) -> str:
    try:
        db = get_db()
        rows = list(db["model_performance"].rows_where(
            "task_type=?",[task_type],order_by="recorded_at desc"))[:20]
        if not rows: return MODEL_CODE
        by_model = {}
        for r in rows:
            m = r["model"]
            if m not in by_model: by_model[m] = []
            by_model[m].append(r["quality"])
        return max(by_model, key=lambda m: sum(by_model[m])/len(by_model[m]))
    except: return MODEL_CODE

# ── RESOURCE MONITOR ─────────────────────────────────────────────────────────
def resource_monitor():
    global RESOURCE_STATUS
    while True:
        try:
            if HAS_PSUTIL:
                ram = psutil.virtual_memory().percent
                cpu = psutil.cpu_percent(interval=1)
            else:
                with open("/proc/meminfo") as f:
                    lines = {l.split(":")[0].strip(): l.split(":")[1].strip()
                             for l in f.readlines()}
                total = int(lines["MemTotal"].split()[0])
                avail = int(lines["MemAvailable"].split()[0])
                ram = round((total-avail)/total*100, 1)
                cpu = 0.0
            RESOURCE_STATUS = {"ram_pct":ram,"cpu_pct":cpu,"warning":ram>RESOURCE_WARN_RAM}
        except: pass
        time.sleep(10)

def can_start_heavy_task() -> bool:
    return RESOURCE_STATUS["ram_pct"] < RESOURCE_WARN_RAM

# ── WEB RESEARCH ──────────────────────────────────────────────────────────────
def search_web(query: str) -> str:
    try:
        r = requests.get(f"{SEARXNG_URL}/search",
            params={"q":query,"format":"json"},timeout=10)
        results = r.json().get("results",[])[:4]
        if not results: return ""
        return "\n".join([f"- {res.get('title','')}: {res.get('content','')[:200]}"
            for res in results])
    except: return ""

def research_task(title: str) -> str:
    try:
        set_agent("Planner","working",f"Recherchiert: {title}")
        results = search_web(title)
        if not results:
            set_agent("Planner","sleeping")
            return ""
        summary = llm(
            f"Aufgabe: {title}\n\nRecherche:\n{results}\n\n"
            "Bester Ansatz? Kostenlose Tools? CPU-only.",
            get_agent_prompt("Planner") or "Technischer Researcher. Nur freie Tools.")
        set_agent("Planner","sleeping")
        return summary
    except Exception as e:
        print(f"research_task error: {e}")
        set_agent("Planner","sleeping")
        return ""

def deep_research(title: str) -> str:
    try:
        set_agent("Planner","working",f"Deep Research: {title}")
        cp = load_checkpoint(-1, f"deep_{title[:40]}")
        if cp:
            set_agent("Planner","sleeping")
            return cp
        r1 = search_web(title)
        if not r1:
            set_agent("Planner","sleeping")
            return ""
        fq_raw = llm(
            f"Aufgabe: {title}\nErste Recherche:\n{r1}\n\n"
            "2-3 spezifische Folge-Suchanfragen? Eine pro Zeile.",
            "Nur Suchanfragen.")
        queries = [q.strip() for q in fq_raw.strip().split('\n')
                   if q.strip() and len(q.strip())>5][:3]
        all_results = [f"[Hauptsuche]\n{r1}"]
        for q in queries:
            r = search_web(q)
            if r: all_results.append(f"[Folge: {q}]\n{r}")
        synthesis = llm(
            f"Aufgabe: {title}\n\n{len(all_results)} Quellen:\n\n" +
            "\n\n---\n\n".join(all_results) +
            "\n\nSynthese: Bester Ansatz, kostenlose Tools, Fallstricke?",
            "Researcher. Synthese aus mehreren Quellen.")
        save_checkpoint(-1, f"deep_{title[:40]}", synthesis)
        set_agent("Planner","sleeping")
        journal_write(f"Deep Research: {title[:80]}",type_="research",author="Planner")
        return synthesis
    except Exception as e:
        print(f"deep_research error: {e}")
        set_agent("Planner","sleeping")
        return ""

# ── THINKING ──────────────────────────────────────────────────────────────────
def get_thinking_mode() -> str:
    try: return get_db()["settings"].get("thinking_mode")["value"]
    except: return "auto"

def set_thinking_mode(mode: str):
    try: get_db()["settings"].upsert({"key":"thinking_mode","value":mode},pk="key")
    except: pass

def should_think_deeply(title: str) -> bool:
    try:
        raw = llm(
            f"Aufgabe: '{title}'\nBraucht diese Aufgabe tiefes Nachdenken? "
            "JA wenn komplex. NUR JA oder NEIN.","Nur JA oder NEIN.")
        return "JA" in raw.strip().upper()
    except: return False

def llm_think(prompt, system, model=None) -> tuple:
    m = model or MODEL_CODE
    try:
        thinking = llm(
            f"Aufgabe: {prompt}\n\n"
            "Denke systematisch nach:\n"
            "〔1〕 Was wird verlangt?\n〔2〕 Teilprobleme?\n"
            "〔3〕 Bester Ansatz?\n〔4〕 Fehlerrisiken?\n〔5〕 Plan:\n",
            system+" Du denkst laut nach.", m)
        answer, tlog = llm_with_tools(
            f"Aufgabe: {prompt}\n\nMeine Analyse:\n{thinking}\n\nFühre jetzt aus:",
            system, m)
        return answer, thinking, tlog
    except Exception as e:
        print(f"llm_think error: {e}")
        answer, tlog = llm_with_tools(prompt, system, m)
        return answer, "", tlog

# ── SKILL SYSTEM ──────────────────────────────────────────────────────────────
def create_or_update_skill(title, result, research, tools_used,
                            feedback="", success=True, quality=7):
    try:
        db = get_db()
        cat  = llm(f"Aufgabe: {title}\nKategorie (1 Wort):","Nur ein Wort.").strip().split()[0][:20]
        name = llm(f"Aufgabe: {title}\nKurzer Skill-Name (2-4 Wörter):","Nur der Name.").strip()[:60]
        existing = list(db["skills"].rows_where("name=?",[name]))
        how_to   = llm(f"Aufgabe: {title}\nLösung: {result[:400]}\nKnappe Anleitung (5-8 Schritte):",
                       "Dokumentationsschreiber.")
        pitfalls = llm(f"Aufgabe: {title}\n2-3 häufige Fehler?","Kurz.")
        tools_str = ", ".join(tools_used) if tools_used else "keine"
        if existing:
            s = existing[0]
            merged = llm(f"Alt:\n{s['how_to']}\n\nNeu:\n{how_to}\n\nBeste Anleitung:",
                         "Verbessere Dokumentation.")
            db["skills"].update(s["id"],{
                "how_to":merged,"pitfalls":(s["pitfalls"] or "")+"\n"+pitfalls,
                "examples":(s["examples"] or "")+f"\n---\nv{s['version']+1}: {title[:80]}",
                "success_count":s["success_count"]+(1 if success else 0),
                "failure_count":s["failure_count"]+(0 if success else 1),
                "version":s["version"]+1,"last_used":datetime.now().isoformat()})
            update_skill_embedding(s["id"], merged)
        else:
            new_id = db["skills"].insert({"name":name,"category":cat,
                "description":f"Gelernt bei: {title[:80]}","how_to":how_to,
                "tools_needed":tools_str,"pitfalls":pitfalls,
                "examples":f"Erstes Beispiel: {title[:80]}","embedding":"",
                "success_count":1 if success else 0,"failure_count":0 if success else 1,
                "version":1,"last_used":datetime.now().isoformat(),
                "created_at":datetime.now().isoformat()}).last_pk
            update_skill_embedding(new_id, how_to)
    except Exception as e:
        print(f"create_or_update_skill error: {e}")

def apply_feedback_to_skill(task_title, feedback):
    try:
        db = get_db()
        skills = list(db["skills"].rows)
        if not skills: return
        overview = "\n".join([f"[{s['id']}] {s['name']}" for s in skills])
        raw = llm(f"Aufgabe: {task_title}\nFeedback: {feedback}\n"
                  f"Skills:\n{overview}\nRelevante ID? Nur Zahl oder 0.","Nur Zahl.")
        sid = int(raw.strip())
        if sid==0: return
        s = db["skills"].get(sid)
        corrected = llm(f"Skill: {s['name']}\nAnleitung:\n{s['how_to']}\n\n"
                       f"Feedback: {feedback}\n\nVerbesserte Anleitung:","Verbessere.")
        db["skills"].update(sid,{"how_to":corrected,
            "failure_count":s["failure_count"]+1,"version":s["version"]+1,
            "last_used":datetime.now().isoformat()})
    except Exception as e:
        print(f"apply_feedback_to_skill error: {e}")

# ── TOOL GAP ANALYSIS ─────────────────────────────────────────────────────────
def analyze_tool_gaps(title, research):
    try:
        db = get_db()
        available = [f"{t['name']}: {t['description']} ({t['status']})" for t in db["tools"].rows]
        raw = llm(
            f"Aufgabe: {title}\nResearch: {research}\nTools:\n{chr(10).join(available)}\n\n"
            "Fehlende kostenlose open-source Tools? JSON:\n"
            '[{"name":"x","reason":"y","install_hint":"z"}] oder []',"Nur JSON.")
        gaps = json.loads(raw.strip().strip("```json").strip("```").strip())
        for g in gaps:
            if not list(db["tool_requests"].rows_where("tool_name=? AND status='pending'",[g["name"]])):
                db["tool_requests"].insert({"tool_name":g["name"],
                    "reason":g.get("reason","Benötigt"),"requested_by":"Agent",
                    "status":"pending","install_hint":g.get("install_hint",""),
                    "created_at":datetime.now().isoformat()})
        return gaps
    except: return []

def get_available_tools_for_task(title):
    try:
        db = get_db()
        available = [t["name"] for t in db["tools"].rows_where("status != 'not_installed'")]
        if not available: return []
        raw = llm(f"Aufgabe: {title}\nTools: {', '.join(available)}\nWelche? NUR JSON oder [].",
                  "Nur JSON.")
        return json.loads(raw.strip().strip("```json").strip("```").strip())
    except: return []

# ── JOURNAL ───────────────────────────────────────────────────────────────────
def journal_write(content, type_="auto", author="Agent", tags=""):
    try:
        get_db()["journal"].insert({"content":content,"type":type_,
            "author":author,"tags":tags,"created_at":datetime.now().isoformat()})
    except: pass

def get_journal_context():
    try:
        entries = list(get_db()["journal"].rows_where(
            order_by="created_at desc"))[:8]
        if not entries: return ""
        return "\n\n--- Journal ---\n" + "\n".join([
            f"[{e['created_at'][:10]} {e['type']}] {e['content'][:150]}"
            for e in reversed(entries)])
    except: return ""

# ── N8N ───────────────────────────────────────────────────────────────────────
def build_n8n_workflow(description):
    try:
        wf_json = llm(
            f"Erstelle n8n Workflow für: {description}\n"
            "NUR valides n8n JSON mit: name, nodes, connections, active:false, settings:{}",
            "n8n-Experte. Nur JSON.")
        clean = wf_json.strip().strip("```json").strip("```").strip()
        wf = json.loads(clean)
        if N8N_API_KEY:
            r = requests.post(f"{N8N_URL}/api/v1/workflows",
                json=wf,headers={"X-N8N-API-KEY":N8N_API_KEY},timeout=30)
            if r.ok:
                return {"ok":True,"created":True,"message":f"Workflow '{wf.get('name')}' erstellt!"}
        return {"ok":True,"created":False,"workflow":clean,
            "message":"JSON generiert. N8N_API_KEY für echte Erstellung setzen."}
    except Exception as e: return {"ok":False,"error":str(e)}

# ── CODE EXECUTION ────────────────────────────────────────────────────────────
def execute_code(code):
    try:
        with tempfile.NamedTemporaryFile(suffix='.py',mode='w',dir='/tmp',delete=False) as f:
            f.write(code); fname=f.name
        result=subprocess.run(['python3',fname],capture_output=True,text=True,
            timeout=30,cwd='/tmp',env={"PATH":"/usr/local/bin:/usr/bin:/bin","HOME":"/tmp"})
        os.unlink(fname)
        return {"stdout":result.stdout[:3000],"stderr":result.stderr[:500],
                "returncode":result.returncode,"success":result.returncode==0}
    except subprocess.TimeoutExpired:
        try: os.unlink(fname)
        except: pass
        return {"stdout":"","stderr":"Timeout (30s)","returncode":-1,"success":False}
    except Exception as e:
        return {"stdout":"","stderr":str(e),"returncode":-1,"success":False}

def extract_and_run_code(text):
    try:
        for pattern in [r'```python\n(.*?)```',r'```\n(.*?)```']:
            m = re.search(pattern,text,re.DOTALL)
            if m: return execute_code(m.group(1).strip())
        if text.strip().startswith(('import ','def ','print(','#!')):
            return execute_code(text.strip())
    except: pass
    return None

# ── AGENT COMMUNICATION ───────────────────────────────────────────────────────
def agent_send(from_a, to_a, task_id, msg_type, content):
    try:
        get_db()["agent_messages"].insert({"from_agent":from_a,"to_agent":to_a,
            "task_id":task_id,"message_type":msg_type,"content":content,
            "status":"unread","created_at":datetime.now().isoformat()})
    except: pass

def agent_recv(agent_name, task_id=None):
    try:
        db = get_db()
        where = "to_agent=? AND status='unread'"
        params = [agent_name]
        if task_id is not None:
            where += " AND task_id=?"; params.append(task_id)
        msgs = list(db["agent_messages"].rows_where(where,params,order_by="created_at"))
        for m in msgs:
            db["agent_messages"].update(m["id"],{"status":"read"})
        return msgs
    except: return []

def agent_msg_context(agent_name, task_id):
    try:
        msgs = agent_recv(agent_name, task_id)
        if not msgs: return ""
        return "\n\n--- Nachrichten ---\n" + "\n".join([
            f"[Von {m['from_agent']}·{m['message_type']}]: {m['content'][:200]}"
            for m in msgs])
    except: return ""

# ── AGENT EVOLUTION ───────────────────────────────────────────────────────────
def evolve_agent_prompt(agent_name, recent_tasks, quality_avg):
    try:
        db = get_db()
        current = db["agent_roster"].get(agent_name)["system_prompt"] or ""
        tasks_text = "\n".join([t["title"][:50] for t in recent_tasks[:5]])
        new_prompt = llm(
            f"Agent: {agent_name}\nAktueller Prompt:\n{current}\n\n"
            f"Durchschnittliche Qualität: {quality_avg:.1f}/10\n"
            f"Zuletzt:\n{tasks_text}\n\nVerbessere den Prompt:",
            "Du verbesserst Agent-System-Prompts.")
        if len(new_prompt) > 50:
            db["agent_roster"].update(agent_name,{"system_prompt":new_prompt.strip()})
            journal_write(f"Agent-Prompt verbessert: {agent_name}",
                type_="evolution",author="Judge")
    except Exception as e:
        print(f"evolve_agent_prompt error: {e}")

# ── HIRE / RETIRE ─────────────────────────────────────────────────────────────
def hire_agent(role, specialization, task_id=None):
    try:
        db = get_db()
        existing = [a["name"] for a in db["agent_roster"].rows_where("role=?",[role])]
        i = 2
        while f"{role.capitalize()}-{i}" in existing: i+=1
        name = f"{role.capitalize()}-{i}"
        try:
            base_prompt = db["agent_roster"].get(role.capitalize())["system_prompt"]
        except:
            base_prompt = ""
        db["agent_roster"].insert({"name":name,"role":role,"specialization":specialization,
            "status":"active","system_prompt":base_prompt,
            "hired_for_task":task_id,"tasks_completed":0,"avg_quality":0.0,
            "hired_at":datetime.now().isoformat(),"retired_at":""})
        set_agent(name,"sleeping")
        journal_write(f"Eingestellt: {name} – {specialization}",type_="hire",author="Chef")
        return name
    except Exception as e:
        print(f"hire_agent error: {e}")
        return "Coder"

def retire_agent(name):
    try:
        db = get_db()
        a = db["agent_roster"].get(name)
        if "-" in name and a["role"] in ("coder","researcher"):
            db["agent_roster"].update(name,{"status":"retired",
                "retired_at":datetime.now().isoformat()})
            try: db["agent_status"].delete_where("name=?",[name])
            except: pass
            journal_write(f"Entlassen: {name}",type_="retire",author="Chef")
    except Exception as e:
        print(f"retire_agent error: {e}")

# ── CHEF ORCHESTRATION ────────────────────────────────────────────────────────
def chef_evaluate(title, research) -> dict:
    try:
        set_agent("Chef","working",f"Bewertet: {title}")
        result = llm(
            f"Aufgabe: {title}\nResearch: {research[:400]}\n\n"
            "Bewerte:\n1. Komplexität 1-10\n2. Anzahl Coder (1-3)\n"
            "3. Cross-Check nötig?\n4. Debate nötig (>=8)?\n"
            "NUR JSON:\n"
            '{"complexity":5,"coder_count":1,"cross_check":false,'
            '"debate":false,"specialization":"","reason":""}',
            get_agent_prompt("Chef") or "Tech-Lead. Nur JSON.")
        set_agent("Chef","sleeping")
        plan = json.loads(result.strip().strip("```json").strip("```").strip())
        plan["coder_count"] = max(1,min(3,int(plan.get("coder_count",1))))
        plan["debate"] = plan.get("complexity",5) >= DEBATE_THRESHOLD
        return plan
    except Exception as e:
        print(f"chef_evaluate error: {e}")
        set_agent("Chef","sleeping")
        return {"complexity":5,"coder_count":1,"cross_check":False,
                "debate":False,"specialization":"","reason":"Standard"}

def chef_final_review(task_id, title, result) -> dict:
    try:
        set_agent("Chef","working",f"Final-Review: {title}")
        review = llm(
            f"Aufgabe: {title}\nLösung:\n{result[:800]}\n\n"
            "Finale Qualitätskontrolle. NUR JSON:\n"
            '{"approved":true,"quality":8,"feedback":"","issues":[]}',
            get_agent_prompt("Chef") or "Tech-Lead. Nur JSON.")
        set_agent("Chef","sleeping")
        return json.loads(review.strip().strip("```json").strip("```").strip())
    except Exception as e:
        print(f"chef_final_review error: {e}")
        set_agent("Chef","sleeping")
        return {"approved":True,"quality":7,"feedback":"OK","issues":[]}

def run_debate(task_id, title, research, skill_ctx, model):
    try:
        set_agent("Coder","working",f"Debatte: {title}")
        set_agent("Critic","working",f"Debatte: {title}")
        pos_a = llm(
            f"Aufgabe: {title}\nResearch: {research[:400]}\n\n"
            "Schlage einen Lösungsansatz vor und begründe ihn (3-4 Sätze):",
            get_agent_prompt("Coder") or "Coder.", model)
        pos_b = llm(
            f"Aufgabe: {title}\nAnsatz:\n{pos_a}\n\n"
            "Stimme zu oder schlage Alternative vor:",
            get_agent_prompt("Critic") or "Critic.", MODEL_CODE)
        consensus = llm(
            f"Aufgabe: {title}\n\nCoder:\n{pos_a}\n\nCritic:\n{pos_b}\n\n"
            "Finaler Plan in 2-3 Sätzen:",
            get_agent_prompt("Chef") or "Chef.")
        set_agent("Coder","sleeping"); set_agent("Critic","sleeping")
        journal_write(f"Debate: {title[:60]}\nKonsens: {consensus[:200]}",
            type_="debate",author="Chef")
        return consensus
    except Exception as e:
        print(f"run_debate error: {e}")
        set_agent("Coder","sleeping"); set_agent("Critic","sleeping")
        return ""

def critic_cross_check(task_id, title, sol_a, sol_b, ag_a, ag_b) -> str:
    try:
        set_agent("Critic","working",f"Cross-Check: {title}")
        synthesis = llm(
            f"Aufgabe: {title}\n\n=== {ag_a} ===\n{sol_a[:600]}\n\n"
            f"=== {ag_b} ===\n{sol_b[:600]}\n\n"
            "Vergleiche und erstelle beste Synthese:",
            get_agent_prompt("Critic") or "Reviewer.")
        set_agent("Critic","sleeping")
        return synthesis
    except Exception as e:
        print(f"critic_cross_check error: {e}")
        set_agent("Critic","sleeping")
        return sol_a

# ── TASK DEPENDENCIES ─────────────────────────────────────────────────────────
def dependencies_met(task_id) -> bool:
    try:
        db = get_db()
        task = db["tasks"].get(task_id)
        deps = json.loads(task.get("depends_on") or "[]")
        if not deps: return True
        for dep_id in deps:
            if db["tasks"].get(dep_id)["status"] != "done": return False
        return True
    except: return True

# ── SLEEP / QUEUE ─────────────────────────────────────────────────────────────
def is_sleep_time():
    try:
        db = get_db()
        schedules = list(db["sleep_schedules"].rows_where("active=1"))
        now = datetime.now(); wd=str(now.weekday()); nt=now.strftime("%H:%M")
        for s in schedules:
            if wd not in s["weekdays"].split(","): continue
            st,en = s["start_time"],s["end_time"]
            if st<=en:
                if st<=nt<=en: return True
            else:
                if nt>=st or nt<=en: return True
    except: pass
    return False

def sleep_watcher():
    global SLEEP_MODE
    while True:
        try:
            sleeping = is_sleep_time()
            if sleeping!=SLEEP_MODE:
                SLEEP_MODE=sleeping
                for name in ["Coder","Critic","Judge","Planner","Chef"]:
                    try:
                        db=get_db()
                        ex=list(db["agent_status"].rows_where("name=?",[name]))
                        if ex and ex[0]["status"] not in ("working","talking"):
                            set_agent(name,"idle" if sleeping else "sleeping")
                    except: pass
                if not sleeping: flush_task_queue()
        except Exception as e: print(f"sleep_watcher: {e}")
        time.sleep(60)

def flush_task_queue():
    global TASK_QUEUE
    with TASK_QUEUE_LOCK:
        q = sorted(TASK_QUEUE,key=lambda x:x[0]); TASK_QUEUE.clear()
    for _,fn,args in q:
        threading.Thread(target=fn,args=args).start()

def queue_or_run(fn, args, priority=2, task_id=None):
    if SLEEP_MODE:
        with TASK_QUEUE_LOCK: TASK_QUEUE.append((priority,fn,args))
        return False
    threading.Thread(target=fn,args=args).start()
    return True

# ── DOCKER ────────────────────────────────────────────────────────────────────
def scan_containers():
    if not docker_client: return
    db=get_db()
    try:
        running=[c.name for c in docker_client.containers.list()]
        all_names=[c.name for c in docker_client.containers.list(all=True)]
        for tool in list(db["tools"].rows):
            pat=tool["container_pattern"]
            mr=[n for n in running if pat in n]
            if mr:
                db["tools"].update(tool["name"],{"container_name":mr[0],
                    "status":"running","updated_at":datetime.now().isoformat()})
            else:
                ma=[n for n in all_names if pat in n]
                db["tools"].update(tool["name"],{"container_name":ma[0] if ma else "",
                    "status":"stopped" if ma else "not_installed",
                    "updated_at":datetime.now().isoformat()})
    except Exception as e: print(f"scan_containers: {e}")

def start_tool(tool_name):
    if not docker_client: return False
    db=get_db()
    try:
        tool=db["tools"].get(tool_name)
        if not tool["container_name"]:
            scan_containers(); tool=db["tools"].get(tool_name)
        if not tool["container_name"]:
            db["tool_requests"].insert({"tool_name":tool_name,
                "reason":"Wird benötigt","requested_by":"Agent","status":"pending",
                "install_hint":tool.get("install_hint",""),
                "created_at":datetime.now().isoformat()})
            return False
        c=docker_client.containers.get(tool["container_name"])
        if c.status!="running": c.start(); time.sleep(3)
        db["tools"].update(tool_name,{"status":"running","updated_at":datetime.now().isoformat()})
        return True
    except Exception as e: print(f"start_tool: {e}"); return False

def stop_tool(tool_name):
    if not docker_client: return
    db=get_db()
    try:
        tool=db["tools"].get(tool_name)
        if not tool["auto_stop"] or not tool["container_name"]: return
        docker_client.containers.get(tool["container_name"]).stop(timeout=5)
        db["tools"].update(tool_name,{"status":"stopped","updated_at":datetime.now().isoformat()})
    except: pass

def get_all_stacks():
    if not docker_client: return []
    try:
        containers=docker_client.containers.list(all=True); stacks={}
        for c in containers:
            s=c.labels.get("com.docker.compose.project","")
            if not s: continue
            if s not in stacks:
                stacks[s]={"name":s,"containers":[],"status":"stopped","running":0,"total":0}
            stacks[s]["containers"].append({"name":c.name,"status":c.status,
                "image":c.image.tags[0] if c.image.tags else str(c.image.id)[:12]})
            stacks[s]["total"]+=1
            if c.status=="running": stacks[s]["running"]+=1
        result=[]
        for name,data in stacks.items():
            if data["running"]==0: data["status"]="stopped"
            elif data["running"]<data["total"]: data["status"]="partial"
            else: data["status"]="running"
            result.append(data)
        return sorted(result,key=lambda x:x["name"])
    except: return []

def stack_action(stack_name, action):
    if not docker_client: return {"ok":False,"error":"Docker nicht verfügbar"}
    try:
        cs=docker_client.containers.list(all=True,filters={
            "label":f"com.docker.compose.project={stack_name}"})
        if not cs: return {"ok":False,"error":f"Stack '{stack_name}' nicht gefunden"}
        for c in cs:
            if action=="start" and c.status!="running": c.start()
            elif action=="stop" and c.status=="running": c.stop(timeout=10)
        return {"ok":True,"affected":len(cs)}
    except Exception as e: return {"ok":False,"error":str(e)}

# ── CORE TASK RUNNER ──────────────────────────────────────────────────────────
def run_task(task_id: int, title: str, priority=2):
    try:
        wait_count = 0
        while not dependencies_met(task_id) and wait_count < 60:
            time.sleep(10); wait_count+=1
        while SLEEP_MODE: time.sleep(30)
        while task_id in PAUSED_TASKS: time.sleep(5)
        db = get_db()
        try:
            task = db["tasks"].get(task_id)
            if task["status"] == "cancelled": return
        except: return

        t_start = time.time()
        db["tasks"].update(task_id,{"status":"running","agent":"Chef",
            "updated_at":datetime.now().isoformat()})

        mode = get_thinking_mode()
        use_thinking = (mode=="on") or (mode=="auto" and should_think_deeply(title))

        cp_research = load_checkpoint(task_id,"research")
        if cp_research:
            research = cp_research
        elif use_thinking:
            research = deep_research(title)
        else:
            research = research_task(title)
        save_checkpoint(task_id,"research",research)

        db["tasks"].update(task_id,{"research":research,"updated_at":datetime.now().isoformat()})

        plan = chef_evaluate(title, research)
        coder_count  = plan["coder_count"]
        cross_check  = plan["cross_check"]
        do_debate    = plan["debate"]
        model = select_model(title)

        skill_ctx = get_relevant_skills(title)
        analyze_tool_gaps(title, research)
        needed = get_available_tools_for_task(title)
        used, tool_note = [], ""
        for t in needed:
            try:
                if start_tool(t): used.append(t); tool_note+=f" [Tool: {t}]"
            except: pass

        approach_note = ""
        if do_debate:
            approach = run_debate(task_id, title, research, skill_ctx, model)
            approach_note = f"\n\nVereinbarter Ansatz:\n{approach}"

        n8n_result = ""
        if "n8n" in title.lower() or "workflow" in title.lower():
            n8n_res = build_n8n_workflow(title)
            if n8n_res.get("ok"):
                n8n_result = f"\n\n[n8n: {n8n_res.get('message','')}]"

        hired = []

        def do_coder_work(agent_name, is_extra=False) -> str:
            try:
                set_agent(agent_name,"working",title)
                msg_ctx = agent_msg_context(agent_name, task_id)
                extra_note = " Deine Lösung wird verglichen." if is_extra else ""
                base = (
                    f"Aufgabe: {title}{tool_note}\nResearch:\n{research}"
                    f"{skill_ctx}{get_journal_context()}{msg_ctx}{approach_note}\n\n"
                    f"Löse vollständig.{extra_note}")
                if use_thinking:
                    sol, thinking, tlog = llm_think(base,
                        get_agent_prompt(agent_name) or "Code-Agent.", model)
                else:
                    sol, tlog = llm_with_tools(base,
                        get_agent_prompt(agent_name) or "Code-Agent.", model)
                set_agent(agent_name,"sleeping")
                return sol
            except Exception as e:
                print(f"do_coder_work error ({agent_name}): {e}")
                set_agent(agent_name,"sleeping")
                return f"Fehler: {e}"

        if coder_count == 1:
            result = do_coder_work("Coder")
        else:
            solutions = {}
            coders = ["Coder"]
            spec = plan.get("specialization","") or f"Spezialist: {title[:40]}"
            for _ in range(coder_count-1):
                na = hire_agent("coder", spec, task_id)
                coders.append(na); hired.append(na)
            threads = []
            for i,c in enumerate(coders):
                def work(cn=c, extra=(i>0)): solutions[cn]=do_coder_work(cn,extra)
                threads.append(threading.Thread(target=work))
            for t in threads: t.start()
            for t in threads: t.join()
            if cross_check and len(solutions)>=2:
                cns = list(solutions.keys())
                result = critic_cross_check(task_id,title,
                    solutions[cns[0]],solutions[cns[1]],cns[0],cns[1])
            else:
                result = solutions.get("Coder",list(solutions.values())[0])

        save_checkpoint(task_id,"result",result)

        while task_id in PAUSED_TASKS: time.sleep(5)
        exec_output=None; exec_summary=""
        exec_result=extract_and_run_code(result)
        if exec_result:
            exec_output=exec_result
            if exec_result["success"]:
                exec_summary=f"\n\n✅ Code ausgeführt:\n{exec_result['stdout'][:400]}"
            else:
                exec_summary=f"\n\n⚠️ Fehler: {exec_result['stderr'][:200]}"
                correction,_ = llm_with_tools(
                    f"Fehler:\n{exec_result['stderr']}\nCode:\n{result[:600]}\nKorrigiere:",
                    "Code-Korrektur.",model)
                r2=extract_and_run_code(correction)
                if r2 and r2["success"]:
                    exec_output=r2; result=correction
                    exec_summary=f"\n\n✅ Korrektur OK:\n{r2['stdout'][:400]}"

        confidence = assess_confidence(title, result, model)
        if confidence < CONFIDENCE_THRESHOLD and coder_count==1:
            extra = hire_agent("coder",f"Zweite Meinung: {title[:40]}",task_id)
            hired.append(extra)
            sol2 = do_coder_work(extra)
            result = critic_cross_check(task_id,title,result,sol2,"Coder",extra)

        if coder_count==1:
            set_agent("Critic","working",f"Prüft: {title}")
            critique,_ = llm_with_tools(
                f"Aufgabe: {title}\nLösung: {result[:500]}\nBewerte kurz (2 Sätze).",
                get_agent_prompt("Critic") or "Reviewer.",MODEL_CODE)
            set_agent("Critic","sleeping")
            result += f"\n\n[Critic: {critique}]"

        chef_review = chef_final_review(task_id, title, result)
        if not chef_review.get("approved",True):
            set_agent("Coder","working",f"Überarbeitung: {title}")
            revision,_ = llm_with_tools(
                f"Aufgabe: {title}\nLösung:\n{result[:500]}\n"
                f"Chef-Feedback: {chef_review.get('feedback','')}\nVerbessere:",
                get_agent_prompt("Coder") or "Code-Agent.",model)
            result=revision; set_agent("Coder","sleeping")

        chef_note = (f"\n\n[Chef ✓ {chef_review.get('quality',7)}/10: "
                     f"{chef_review.get('feedback','OK')}]")
        final = result + exec_summary + n8n_result + chef_note
        duration = time.time() - t_start
        quality = chef_review.get("quality",7)

        db["tasks"].update(task_id,{"status":"done","result":final,
            "skills_used":", ".join(used),"model_used":f"{model}×{coder_count}",
            "confidence":confidence,
            "exec_output":json.dumps(exec_output) if exec_output else "",
            "updated_at":datetime.now().isoformat()})

        record_model_performance(model,"code",quality,True,duration)
        set_agent("Chef","sleeping")
        set_agent("Judge","working",f"Skill: {title}")
        create_or_update_skill(title,result,research,used,success=True,quality=quality)
        journal_write(f"Erledigt ({coder_count} Coder, {confidence}/10): {title[:80]}",
            type_="auto",author="Chef")
        set_agent("Judge","sleeping")

        done_count = db["tasks"].count_where("status='done'")
        if done_count % 10 == 0:
            recent = list(db["tasks"].rows_where("status='done'",order_by="updated_at desc"))[:10]
            for aname in ["Coder","Critic","Planner"]:
                threading.Thread(target=evolve_agent_prompt,args=(aname,recent,quality)).start()

        for a in hired:
            try:
                db["agent_roster"].update(a,{"tasks_completed":
                    db["agent_roster"].get(a)["tasks_completed"]+1})
            except: pass
            retire_agent(a)
        for t in set(used): stop_tool(t)
        check_self_improve()

    except Exception as e:
        print(f"run_task error (task {task_id}): {e}")
        try:
            get_db()["tasks"].update(task_id,{"status":"failed",
                "result":f"Fehler: {e}","updated_at":datetime.now().isoformat()})
        except: pass

def run_file_task(task_id, title, filename, ftype):
    try:
        while SLEEP_MODE: time.sleep(30)
        while task_id in PAUSED_TASKS: time.sleep(5)
        db=get_db()
        if db["tasks"].get(task_id)["status"]=="cancelled": return
        set_agent("Coder","working",f"Datei: {filename}")
        db["tasks"].update(task_id,{"status":"running","agent":"Coder",
            "updated_at":datetime.now().isoformat()})
        model=select_model(title,ftype)
        research=research_task(f"{title} ({ftype})")
        skill_ctx=get_relevant_skills(title)
        analyze_tool_gaps(title,research)
        needed=get_available_tools_for_task(title+f" {ftype}")
        used,tool_note=[],""
        for t in needed:
            try:
                if start_tool(t): used.append(t); tool_note+=f" [Tool: {t}]"
            except: pass
        wp=WORKSPACE/filename
        if ftype=="video":
            outname=f"output_{filename}"; outpath=OUTBOX/outname
            cmd=llm(f"Aufgabe: {title}\nEingabe: {wp}\nAusgabe: {outpath}\n{skill_ctx}\nNUR FFmpeg-Befehl.","FFmpeg-Experte.")
            cmd=cmd.strip().strip("```").strip()
            if cmd.startswith("ffmpeg"):
                r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=300)
                result=f"Video OK: {outname}" if r.returncode==0 else f"FFmpeg-Fehler: {r.stderr[:400]}"
            else: result=f"Kein gültiger Befehl: {cmd[:200]}"
        elif ftype=="audio":
            result=f"Audio: {filename}"
            try:
                if start_tool("whisper-asr"):
                    r=requests.post("http://localhost:9000/asr",
                        files={"audio_file":open(wp,"rb")},
                        params={"task":"transcribe","language":"de"},timeout=1800)
                    if r.ok:
                        text=r.json().get("text",""); on=wp.stem+"_transkript.txt"
                        (OUTBOX/on).write_text(text); result=f"Transkription: {on}\n{text[:200]}..."
            except: pass
        elif ftype=="image":
            try: result=f"Bild-Analyse:\n{llm_vision(f'Beschreibe detailliert. Aufgabe: {title}',str(wp))}"
            except: result=llm(f"Bildanalyse {filename}: {title}. LLaVA nicht verfügbar.")
        else:
            result,_ = llm_with_tools(
                f"Aufgabe: {title}\nDatei: {filename} ({ftype}){tool_note}\n"
                f"Research: {research}\n{skill_ctx}\nVerarbeite vollständig.",
                get_agent_prompt("Coder") or "Code-Agent.", model)
        exec_result=extract_and_run_code(result)
        exec_note=""
        if exec_result:
            exec_note=f"\n\n✅ {exec_result['stdout'][:300]}" \
                if exec_result["success"] else f"\n\n⚠️ {exec_result['stderr'][:150]}"
        outfile=""
        for c in [OUTBOX/f"output_{filename}",OUTBOX/filename]:
            if c.exists(): outfile=c.name; break
        confidence=assess_confidence(title,result,model)
        db["tasks"].update(task_id,{"status":"done","result":result+exec_note,
            "output_file":outfile,"model_used":model,"confidence":confidence,
            "updated_at":datetime.now().isoformat()})
        set_agent("Coder","sleeping")
        set_agent("Judge","working",f"Skill: {title}")
        create_or_update_skill(title,result,research,used)
        set_agent("Judge","sleeping")
        for t in set(used): stop_tool(t)
        check_self_improve()
    except Exception as e:
        print(f"run_file_task error: {e}")
        try:
            get_db()["tasks"].update(task_id,{"status":"failed",
                "result":f"Fehler: {e}","updated_at":datetime.now().isoformat()})
        except: pass

def plan_project(project_id, name, description, template_steps=None):
    try:
        set_agent("Planner","working",f"Plant: {name}")
        db=get_db()
        if template_steps:
            steps=[{"title":s} for s in template_steps]
        else:
            research=research_task(name+" "+description)
            plan_text=llm(
                f"Plan für: {name}\nBeschreibung: {description}\nResearch: {research}\n"
                "NUR JSON, max 5 Schritte:\n[{\"title\":\"Schritt 1\"}]",
                get_agent_prompt("Planner") or "Projektplaner. Nur JSON.")
            try:
                steps=json.loads(plan_text.strip().strip("```json").strip("```").strip())
            except:
                steps=[{"title":"Analysieren"},{"title":"Implementieren"},{"title":"Prüfen"}]
        for i,step in enumerate(steps[:5]):
            db["project_steps"].insert({"project_id":project_id,"step_num":i+1,
                "title":step.get("title",f"Schritt {i+1}"),
                "status":"pending","result":"","updated_at":datetime.now().isoformat()})
        set_agent("Planner","sleeping")
        execute_project(project_id)
    except Exception as e:
        print(f"plan_project error: {e}")
        set_agent("Planner","sleeping")
        try:
            get_db()["projects"].update(project_id,{"status":"failed",
                "updated_at":datetime.now().isoformat()})
        except: pass

def execute_project(project_id):
    try:
        db=get_db()
        project=db["projects"].get(project_id)
        steps=list(db["project_steps"].rows_where(
            "project_id=? AND status='pending'",[project_id],order_by="step_num"))
        hired=[]
        for step in steps:
            while SLEEP_MODE: time.sleep(30)
            if db["projects"].get(project_id)["status"]=="cancelled": break
            set_agent("Coder","working",step["title"])
            db["project_steps"].update(step["id"],{"status":"running","updated_at":datetime.now().isoformat()})
            db["projects"].update(project_id,{"status":"running","updated_at":datetime.now().isoformat()})
            skill_ctx=get_relevant_skills(step["title"])
            model=select_model(step["title"])
            needed=get_available_tools_for_task(step["title"])
            used,tool_note=[],""
            for t in needed:
                try:
                    if start_tool(t): used.append(t); tool_note+=f" [Tool: {t}]"
                except: pass
            result,tlog=llm_with_tools(
                f"Projekt: {project['name']}\nAufgabe: {step['title']}{tool_note}\n"
                f"{skill_ctx}\nFühre vollständig durch.",
                get_agent_prompt("Coder") or "Code-Agent.",model)
            exec_result=extract_and_run_code(result)
            exec_note=""
            if exec_result:
                exec_note=f"\n✅ {exec_result['stdout'][:200]}" \
                    if exec_result["success"] else f"\n⚠️ {exec_result['stderr'][:100]}"
            db["project_steps"].update(step["id"],{"status":"done",
                "result":result+exec_note,"updated_at":datetime.now().isoformat()})
            set_agent("Coder","sleeping")
            set_agent("Critic","working",f"Prüft: {step['title']}")
            critique,_=llm_with_tools(
                f"Bewerte (2 Sätze): {step['title']}\nLösung: {result[:400]}",
                get_agent_prompt("Critic") or "Reviewer.",MODEL_CODE)
            db["project_steps"].update(step["id"],
                {"result":result+exec_note+f"\n\nCritic: {critique}",
                 "updated_at":datetime.now().isoformat()})
            set_agent("Critic","sleeping")
            set_agent("Judge","working",f"Skill: {step['title']}")
            create_or_update_skill(step["title"],result,"",used)
            set_agent("Judge","sleeping")
        for t in set(used): stop_tool(t)
        final="cancelled" if db["projects"].get(project_id)["status"]=="cancelled" else "done"
        db["projects"].update(project_id,{"status":final,"updated_at":datetime.now().isoformat()})
        journal_write(f"Projekt abgeschlossen: {project['name']}",type_="auto",author="Chef")
        check_self_improve()
    except Exception as e:
        print(f"execute_project error: {e}")
        try:
            get_db()["projects"].update(project_id,{"status":"failed",
                "updated_at":datetime.now().isoformat()})
        except: pass

def process_inbox_file(filepath):
    try:
        db=get_db(); filename=filepath.name
        if list(db["inbox_log"].rows_where("filename=?",[filename])): return
        db["inbox_log"].insert({"filename":filename,"processed_at":datetime.now().isoformat()})
        if filepath.suffix.lower()==".txt": return
        instr_file=filepath.with_suffix(".txt")
        instruction=instr_file.read_text(encoding="utf-8",errors="ignore").strip() \
            if instr_file.exists() else ""
        ftype=file_type(filepath)
        if not instruction:
            defaults={"video":"Schneide und optimiere","audio":"Transkribiere",
                      "image":"Analysiere detailliert","code":"Überprüfe und verbessere",
                      "text":"Analysiere"}
            instruction=defaults.get(ftype,f"Verarbeite: {filename}")
        shutil.copy2(filepath,WORKSPACE/filename)
        row_id=db["tasks"].insert({"project_id":None,"title":instruction,
            "status":"pending","priority":2,"paused":0,"agent":"","result":"",
            "research":"","skills_used":"","feedback":"","source_file":filename,
            "output_file":"","model_used":"","exec_output":"","confidence":0,
            "depends_on":"[]","created_at":datetime.now().isoformat(),
            "updated_at":datetime.now().isoformat()}).last_pk
        queue_or_run(run_file_task,(row_id,instruction,filename,ftype),priority=2)
    except Exception as e:
        print(f"process_inbox_file error: {e}")

def inbox_watcher():
    while True:
        try:
            if not SLEEP_MODE:
                for f in INBOX.iterdir():
                    if f.is_file() and f.suffix.lower()!=".txt":
                        process_inbox_file(f)
        except Exception as e: print(f"inbox_watcher: {e}")
        time.sleep(10)

def check_self_improve():
    try:
        db=get_db()
        total=db["tasks"].count_where("status='done'")+db["projects"].count_where("status='done'")
        improvements=db["improvements"].count
        if total>0 and total%IMPROVE_AFTER==0 and improvements<total//IMPROVE_AFTER:
            threading.Thread(target=self_improve).start()
    except: pass

def self_improve():
    try:
        set_agent("Judge","working","Selbstverbesserung...")
        db=get_db()
        skills=list(db["skills"].rows)
        skill_summary="\n".join([f"- {s['name']} v{s['version']}"
            for s in skills]) if skills else "Keine Skills"
        recent=list(db["tasks"].rows_where("status='done'",order_by="updated_at desc"))[:8]
        tasks_text="\n".join([t["title"] for t in recent])
        analysis=llm(
            f"Skills:\n{skill_summary}\n\nZuletzt:\n{tasks_text}\n\n"
            "Verbesserungsvorschläge? Nur kostenlose, self-hostbare Ansätze.",
            get_agent_prompt("Judge") or "KI-System das sich verbessert.")
        db["improvements"].insert({"summary":analysis,"applied_at":datetime.now().isoformat()})
        journal_write(f"Selbstverbesserung:\n{analysis}",type_="improvement",author="Judge")
        agent_propose_improvement()
        set_agent("Judge","sleeping")
    except Exception as e:
        print(f"self_improve error: {e}")
        set_agent("Judge","sleeping")

def agent_propose_improvement():
    try:
        db=get_db()
        with open("/app/backend/main.py","r") as f: code=f.read()
        recent=list(db["tasks"].rows_where("status='done'",order_by="updated_at desc"))[:5]
        proposal=llm(
            f"Code (Auszug):\n{code[:2000]}\n\nAufgaben:\n"
            f"{chr(10).join([t['title'] for t in recent])}\n\n"
            'Verbesserungsvorschlag. NUR JSON: {"title":"X","description":"Y","new_code":""}',
            "Python-Entwickler. new_code leer.")
        prop=json.loads(proposal.strip().strip("```json").strip("```").strip())
        if prop.get("title") and db["code_updates"].count_where("status='pending'")<5:
            db["code_updates"].insert({"title":prop["title"],
                "description":prop.get("description",""),
                "new_code":prop.get("new_code",""),"status":"pending",
                "proposed_by":"Self-Improve Agent",
                "created_at":datetime.now().isoformat(),"reviewed_at":""})
    except Exception as e:
        print(f"agent_propose_improvement error: {e}")

def proactive_code_analyst():
    time.sleep(3600)
    while True:
        try:
            db=get_db()
            if (db["tasks"].count_where("status='done'")>=5 and
                    db["code_updates"].count_where("status='pending'")<3):
                set_agent("Judge","working","Proaktive Code-Analyse...")
                agent_propose_improvement()
                set_agent("Judge","sleeping")
        except Exception as e: print(f"proactive_analyst: {e}")
        time.sleep(7200)

def apply_and_restart(uid):
    time.sleep(1)
    db=get_db()
    try:
        pending="/memory/pending_update.py"
        if not os.path.exists(pending): return
        with open(pending,"r") as f: code=f.read()
        compile(code,"main.py","exec")
        with open("/app/backend/main.py","w") as f: f.write(code)
        os.remove(pending)
        if docker_client:
            try: docker_client.containers.get("agentoffice-backend").restart(); return
            except: pass
        os.execv(sys.executable,[sys.executable]+sys.argv)
    except SyntaxError as e:
        db["code_updates"].update(uid,{"status":f"failed: {e}",
            "reviewed_at":datetime.now().isoformat()})

def do_ui_modify(instruction):
    try:
        set_agent("Judge","working",f"UI: {instruction}")
        src=FRONTEND_PATH if os.path.exists(FRONTEND_PATH) else FRONTEND_SRC
        try:
            with open(src,"r") as f: current=f.read()
        except: current=""
        new_html=llm(f"HTML:\n{current[:3000]}\n\nÄnderung: {instruction}\n\nNUR vollständiger HTML-Code.",
                     "Frontend-Entwickler. Nur HTML.")
        clean=new_html.strip().strip("```html").strip("```").strip()
        if "<html" in clean or "<!DOCTYPE" in clean:
            with open(FRONTEND_PATH,"w") as f: f.write(clean)
        set_agent("Judge","sleeping")
    except Exception as e:
        print(f"do_ui_modify error: {e}")
        set_agent("Judge","sleeping")

# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/",response_class=HTMLResponse)
def root():
    if os.path.exists(FRONTEND_PATH): return FileResponse(FRONTEND_PATH)
    if os.path.exists(FRONTEND_SRC):
        shutil.copy(FRONTEND_SRC,FRONTEND_PATH); return FileResponse(FRONTEND_PATH)
    return HTMLResponse("<h1>Agent Office startet...</h1>")

@app.get("/status")
def get_status():
    db=get_db()
    return {"agents":list(db["agent_status"].rows),
        "active_tasks":list(db["tasks"].rows_where("status='running'")),
        "sleep_mode":SLEEP_MODE,"queued_tasks":len(TASK_QUEUE),
        "pending_updates":db["code_updates"].count_where("status='pending'"),
        "pending_tool_requests":db["tool_requests"].count_where("status='pending'"),
        "skills_count":db["skills"].count,"resources":RESOURCE_STATUS}

@app.get("/tasks")
def get_tasks(): return list(get_db()["tasks"].rows)

@app.get("/tasks/summary")
def get_tasks_summary():
    db=get_db()
    pending=list(db["tasks"].rows_where("status in ('pending','queued','running')"))
    if len(pending)>7:
        summary=llm(
            "Aufgaben:\n"+"\n".join([f"- {t['title'][:50]}" for t in pending])+
            "\n\nKurze Zusammenfassung?","Projektmanager.")
        return {"summary":summary,"pending_count":len(pending)}
    return {"summary":"","pending_count":len(pending)}

@app.post("/task")
def add_task(task: dict):
    db=get_db()
    priority=task.get("priority",2)
    row_id=db["tasks"].insert({"project_id":task.get("project_id"),
        "title":task["title"],"status":"pending","priority":priority,"paused":0,
        "agent":"","result":"","research":"","skills_used":"","feedback":"",
        "source_file":"","output_file":"","model_used":"","exec_output":"",
        "confidence":0,"depends_on":json.dumps(task.get("depends_on",[])),
        "created_at":datetime.now().isoformat(),
        "updated_at":datetime.now().isoformat()}).last_pk
    queued=not queue_or_run(run_task,(row_id,task["title"],priority),priority=priority)
    if queued:
        db["tasks"].update(row_id,{"status":"queued","updated_at":datetime.now().isoformat()})
    return {"ok":True,"id":row_id,"queued":queued}

@app.post("/tasks/{tid}/pause")
def pause_task(tid: int):
    db=get_db()
    task=db["tasks"].get(tid)
    if task["paused"]:
        PAUSED_TASKS.discard(tid)
        db["tasks"].update(tid,{"paused":0,"updated_at":datetime.now().isoformat()})
        return {"ok":True,"paused":False}
    else:
        PAUSED_TASKS.add(tid)
        db["tasks"].update(tid,{"paused":1,"updated_at":datetime.now().isoformat()})
        return {"ok":True,"paused":True}

@app.post("/tasks/{tid}/cancel")
def cancel_task(tid: int):
    PAUSED_TASKS.discard(tid)
    get_db()["tasks"].update(tid,{"status":"cancelled","paused":0,
        "updated_at":datetime.now().isoformat()})
    return {"ok":True}

@app.post("/tasks/{tid}/priority")
def set_priority(tid: int, data: dict):
    get_db()["tasks"].update(tid,{"priority":max(1,min(3,int(data.get("priority",2)))),
        "updated_at":datetime.now().isoformat()})
    return {"ok":True}

@app.post("/tasks/{tid}/feedback")
def task_feedback(tid: int, data: dict):
    db=get_db(); feedback=data.get("feedback",""); rating=data.get("rating","ok")
    try:
        task=db["tasks"].get(tid)
        db["tasks"].update(tid,{"feedback":feedback,"updated_at":datetime.now().isoformat()})
        if rating in ("bad","wrong") and task["title"]:
            threading.Thread(target=apply_feedback_to_skill,
                args=(task["title"],feedback)).start()
    except: pass
    return {"ok":True}

@app.get("/projects")
def get_projects():
    db=get_db()
    projects=list(db["projects"].rows)
    for p in projects:
        steps=list(db["project_steps"].rows_where("project_id=?",[p["id"]],order_by="step_num"))
        p["steps"]=steps
        p["progress"]=len([s for s in steps if s["status"]=="done"])
        p["total_steps"]=len(steps)
    return projects

@app.post("/project")
def create_project(data: dict):
    db=get_db()
    pid=db["projects"].insert({"name":data["name"],"description":data.get("description",""),
        "status":"planning","created_at":datetime.now().isoformat(),
        "updated_at":datetime.now().isoformat()}).last_pk
    queue_or_run(plan_project,(pid,data["name"],data.get("description",""),
                               data.get("template_steps")))
    return {"ok":True,"id":pid}

@app.post("/project/{pid}/cancel")
def cancel_project(pid: int):
    db=get_db()
    db["projects"].update(pid,{"status":"cancelled","updated_at":datetime.now().isoformat()})
    # FIX: update_where ersetzt durch Loop
    for step in list(db["project_steps"].rows_where(
            "project_id=? AND status='pending'",[pid])):
        db["project_steps"].update(step["id"],{"status":"cancelled"})
    return {"ok":True}

@app.get("/templates")
def get_templates():
    db=get_db()
    ts=list(db["project_templates"].rows)
    for t in ts:
        try: t["steps"]=json.loads(t["steps"])
        except: t["steps"]=[]
    return ts

@app.post("/templates")
def add_template(data: dict):
    get_db()["project_templates"].insert({"name":data["name"],
        "description":data.get("description",""),"category":data.get("category","general"),
        "steps":json.dumps(data.get("steps",[])),"created_at":datetime.now().isoformat()})
    return {"ok":True}

@app.delete("/templates/{tid}")
def delete_template(tid: int): get_db()["project_templates"].delete(tid); return {"ok":True}

@app.get("/quick-tasks")
def get_quick_tasks(): return QUICK_TASKS

@app.get("/skills")
def get_skills(): return list(get_db()["skills"].rows)

@app.put("/skills/{sid}")
def update_skill(sid: int, data: dict):
    db=get_db(); update={}
    for f in ["name","description","how_to","tools_needed","pitfalls","category"]:
        if f in data: update[f]=data[f]
    if update:
        update["version"]=db["skills"].get(sid)["version"]+1
        db["skills"].update(sid,update)
        update_skill_embedding(sid,update.get("how_to",""))
    return {"ok":True}

@app.delete("/skills/{sid}")
def delete_skill(sid: int): get_db()["skills"].delete(sid); return {"ok":True}

@app.get("/improvements")
def get_improvements(): return list(get_db()["improvements"].rows)

@app.get("/model-performance")
def get_model_performance():
    try:
        db=get_db()
        rows=list(db["model_performance"].rows_where(order_by="recorded_at desc"))[:100]
        by_model={}
        for r in rows:
            key=f"{r['model']}::{r['task_type']}"
            if key not in by_model: by_model[key]={"model":r["model"],"task_type":r["task_type"],"qualities":[],"count":0}
            by_model[key]["qualities"].append(r["quality"])
            by_model[key]["count"]+=1
        result=[]
        for k,v in by_model.items():
            v["avg_quality"]=round(sum(v["qualities"])/len(v["qualities"]),1)
            del v["qualities"]; result.append(v)
        return sorted(result,key=lambda x:-x["avg_quality"])
    except: return []

@app.get("/journal")
def get_journal():
    return list(get_db()["journal"].rows_where(order_by="created_at desc"))

@app.post("/journal")
def add_journal_entry(data: dict):
    get_db()["journal"].insert({"content":data["content"],
        "type":data.get("type","manual"),"author":data.get("author","Du"),
        "tags":data.get("tags",""),"created_at":datetime.now().isoformat()})
    return {"ok":True}

@app.delete("/journal/{jid}")
def delete_journal_entry(jid: int): get_db()["journal"].delete(jid); return {"ok":True}

@app.post("/execute")
def execute_endpoint(data: dict):
    code=data.get("code","")
    if not code: return {"ok":False,"error":"Kein Code"}
    return {"ok":True,"result":execute_code(code)}

@app.post("/chat")
def chat(data: dict):
    try:
        messages=data.get("messages",[])
        db=get_db()
        running=list(db["tasks"].rows_where("status='running'"))
        pending=list(db["tasks"].rows_where("status in ('pending','queued')"))
        projects=list(db["projects"].rows_where("status in ('running','planning')"))
        done=list(db["tasks"].rows_where("status='done'",order_by="updated_at desc"))[:3]
        context=(
            f"Du bist der Chef-Agent von Agent Office.\n"
            f"Status: {len(running)} laufend, {len(pending)} wartend, "
            f"{len(projects)} Projekte aktiv.\n"
            f"Laufend: {', '.join([t['title'][:40] for t in running])}\n"
            f"Wartend: {', '.join([t['title'][:30] for t in pending[:5]])}\n"
            f"Zuletzt: {', '.join([t['title'][:40] for t in done])}\n"
            f"RAM: {RESOURCE_STATUS['ram_pct']}%, Thinking: {get_thinking_mode()}\n"
            f"{get_journal_context()}\n"
            "Antworte auf Deutsch. Sei direkt und hilfreich."
        )
        conv="\n".join([f"{'Nutzer' if m['role']=='user' else 'Chef'}: {m['content']}"
            for m in messages[-8:]])
        response=llm(conv,context)
        return {"response":response}
    except Exception as e:
        return {"response":f"Fehler: {e}"}

@app.get("/agents/roster")
def get_roster():
    return list(get_db()["agent_roster"].rows_where(order_by="hired_at"))

@app.get("/agents/messages")
def get_messages(task_id: int=None):
    db=get_db()
    if task_id:
        return list(db["agent_messages"].rows_where("task_id=?",[task_id],
            order_by="created_at desc"))[:50]
    return list(db["agent_messages"].rows_where(order_by="created_at desc"))[:100]

@app.post("/agents/hire")
def manual_hire(data: dict):
    name=hire_agent(data.get("role","coder"),data.get("specialization","Allgemein"))
    return {"ok":True,"name":name}

@app.post("/agents/{name}/retire")
def manual_retire(name: str): retire_agent(name); return {"ok":True}

@app.put("/agents/{name}/prompt")
def update_agent_prompt(name: str, data: dict):
    try: get_db()["agent_roster"].update(name,{"system_prompt":data.get("prompt","")})
    except: pass
    return {"ok":True}

@app.get("/resources")
def get_resources(): return RESOURCE_STATUS

@app.get("/settings")
def get_settings():
    try: return {"thinking_mode":get_db()["settings"].get("thinking_mode")["value"]}
    except: return {"thinking_mode":"auto"}

@app.post("/settings")
def update_settings(data: dict):
    if "thinking_mode" in data:
        mode=data["thinking_mode"]
        if mode in ("on","off","auto"): set_thinking_mode(mode)
    return {"ok":True}

@app.get("/schedules")
def get_schedules(): return list(get_db()["sleep_schedules"].rows)

@app.post("/schedules")
def add_schedule(data: dict):
    get_db()["sleep_schedules"].insert({"name":data["name"],
        "weekdays":data["weekdays"],"start_time":data["start_time"],
        "end_time":data["end_time"],"active":int(data.get("active",True)),
        "created_at":datetime.now().isoformat()})
    return {"ok":True}

@app.delete("/schedules/{sid}")
def delete_schedule(sid: int): get_db()["sleep_schedules"].delete(sid); return {"ok":True}

@app.post("/schedules/{sid}/toggle")
def toggle_schedule(sid: int):
    db=get_db(); s=db["sleep_schedules"].get(sid)
    db["sleep_schedules"].update(sid,{"active":1-s["active"]}); return {"ok":True}

@app.get("/stacks")
def get_stacks(): return get_all_stacks()

@app.post("/stacks/{name}/start")
def start_stack(name: str): return stack_action(name,"start")

@app.post("/stacks/{name}/stop")
def stop_stack(name: str): return stack_action(name,"stop")

@app.get("/tools")
def get_tools(): scan_containers(); return list(get_db()["tools"].rows)

@app.post("/tools")
def add_tool(data: dict):
    get_db()["tools"].upsert({"name":data["name"],"description":data.get("description",""),
        "api_url":data.get("api_url",""),"container_name":"",
        "stack_name":data.get("stack_name",""),"status":"unknown",
        "auto_stop":int(data.get("auto_stop",True)),
        "container_pattern":data.get("container_pattern",data["name"]),
        "install_hint":data.get("install_hint",""),
        "updated_at":datetime.now().isoformat()},pk="name")
    return {"ok":True}

@app.post("/tools/{name}/start")
def api_start_tool(name: str): return {"ok":start_tool(name)}

@app.post("/tools/{name}/stop")
def api_stop_tool(name: str): stop_tool(name); return {"ok":True}

@app.get("/tool-requests")
def get_tool_requests(): return list(get_db()["tool_requests"].rows)

@app.post("/tool-requests/{rid}/approve")
def approve_request(rid: int):
    get_db()["tool_requests"].update(rid,{"status":"approved"}); return {"ok":True}

@app.post("/tool-requests/{rid}/reject")
def reject_request(rid: int):
    get_db()["tool_requests"].update(rid,{"status":"rejected"}); return {"ok":True}

@app.get("/updates")
def get_updates(): return list(get_db()["code_updates"].rows)

@app.post("/updates/propose")
def propose_update(data: dict):
    get_db()["code_updates"].insert({"title":data["title"],
        "description":data.get("description",""),"new_code":data.get("new_code",""),
        "status":"pending","proposed_by":data.get("proposed_by","Du"),
        "created_at":datetime.now().isoformat(),"reviewed_at":""})
    return {"ok":True}

@app.post("/updates/{uid}/reject")
def reject_update(uid: int):
    get_db()["code_updates"].update(uid,{"status":"rejected",
        "reviewed_at":datetime.now().isoformat()}); return {"ok":True}

@app.post("/updates/{uid}/approve")
def approve_update(uid: int):
    db=get_db(); update=db["code_updates"].get(uid)
    new_code=update["new_code"]
    if not new_code or len(new_code)<100: return {"ok":False,"error":"Code zu kurz"}
    try:
        with open("/app/backend/main.py","r") as f:
            with open("/memory/main_backup.py","w") as b: b.write(f.read())
        with open("/memory/pending_update.py","w") as f: f.write(new_code)
        db["code_updates"].update(uid,{"status":"approved",
            "reviewed_at":datetime.now().isoformat()})
        threading.Thread(target=apply_and_restart,args=(uid,)).start()
        return {"ok":True}
    except Exception as e: return {"ok":False,"error":str(e)}

@app.post("/ui/modify")
def modify_ui(data: dict):
    instruction=data.get("instruction","")
    if not instruction: return {"ok":False}
    threading.Thread(target=do_ui_modify,args=(instruction,)).start()
    return {"ok":True}

@app.get("/files/inbox")
def list_inbox():
    db=get_db(); files=[]
    for f in INBOX.iterdir():
        if f.is_file():
            processed=bool(list(db["inbox_log"].rows_where("filename=?",[f.name])))
            files.append({"name":f.name,"size":f.stat().st_size,
                "type":file_type(f),"processed":processed,
                "modified":datetime.fromtimestamp(f.stat().st_mtime).isoformat()})
    return sorted(files,key=lambda x:x["modified"],reverse=True)

@app.get("/files/outbox")
def list_outbox():
    return sorted([{"name":f.name,"size":f.stat().st_size,"type":file_type(f),
        "modified":datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
        for f in OUTBOX.iterdir() if f.is_file()],
        key=lambda x:x["modified"],reverse=True)

@app.get("/files/workspace")
def list_workspace():
    return sorted([{"name":f.name,"size":f.stat().st_size,"type":file_type(f),
        "modified":datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
        for f in WORKSPACE.iterdir() if f.is_file()],
        key=lambda x:x["modified"],reverse=True)

@app.post("/files/upload")
async def upload_file(file: UploadFile=File(...), instruction: str=""):
    dest=INBOX/file.filename
    with open(dest,"wb") as f: shutil.copyfileobj(file.file,f)
    if instruction: (INBOX/(Path(file.filename).stem+".txt")).write_text(instruction)
    return {"ok":True,"filename":file.filename}

@app.get("/files/outbox/{filename}")
def download_outbox(filename: str):
    fp=OUTBOX/filename
    if fp.exists(): return FileResponse(fp,filename=filename)
    return {"error":"Nicht gefunden"}

@app.delete("/files/outbox/{filename}")
def delete_outbox_file(filename: str):
    f=OUTBOX/filename
    if f.exists(): f.unlink()
    return {"ok":True}

@app.delete("/files/inbox/{filename}")
def delete_inbox_file(filename: str):
    for f in [INBOX/filename,INBOX/(Path(filename).stem+".txt")]:
        if f.exists(): f.unlink()
    return {"ok":True}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            db=get_db()
            await websocket.send_text(json.dumps({
                "agents":list(db["agent_status"].rows),
                "sleep_mode":SLEEP_MODE,"queued_tasks":len(TASK_QUEUE),
                "pending_updates":db["code_updates"].count_where("status='pending'"),
                "pending_tool_requests":db["tool_requests"].count_where("status='pending'"),
                "skills_count":db["skills"].count,
                "paused_tasks":list(PAUSED_TASKS),
                "resources":RESOURCE_STATUS}))
            await asyncio.sleep(2)
        except: break

if __name__=="__main__":
    os.makedirs("/memory",exist_ok=True)
    INBOX.mkdir(exist_ok=True); OUTBOX.mkdir(exist_ok=True); WORKSPACE.mkdir(exist_ok=True)
    if not os.path.exists(FRONTEND_PATH) and os.path.exists(FRONTEND_SRC):
        shutil.copy(FRONTEND_SRC,FRONTEND_PATH)
    threading.Thread(target=scan_containers,daemon=True).start()
    threading.Thread(target=inbox_watcher,daemon=True).start()
    threading.Thread(target=sleep_watcher,daemon=True).start()
    threading.Thread(target=proactive_code_analyst,daemon=True).start()
    threading.Thread(target=resource_monitor,daemon=True).start()
    uvicorn.run(app,host="0.0.0.0",port=8000)
