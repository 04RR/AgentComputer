"""
Phase 3 smoke test: manage_personas tool, cron CRUD, model overrides.
Run: python test_phase3.py
"""
import requests
import json
import sys
import time

BASE = "http://localhost:8000"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}ℹ{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")

errors = 0

def check(label, condition, detail=""):
    global errors
    if condition:
        ok(label)
    else:
        fail(f"{label} — {detail}")
        errors += 1

print(f"{BOLD}{'='*60}")
print("Phase 3 Smoke Test")
print(f"{'='*60}{RESET}\n")

# ── Cleanup from previous runs ──
requests.delete(f"{BASE}/api/personas/news-scout")
requests.delete(f"{BASE}/api/personas/cron-test")
requests.delete(f"{BASE}/api/sessions/phase3-chat-test")

# ══════════════════════════════════════════════════════════════
# Feature 1: manage_personas tool
# ══════════════════════════════════════════════════════════════
print(f"{BOLD}Feature 1: manage_personas tool{RESET}\n")

# 1A. Check the tool is registered
print(f"{BOLD}Test 1A: Tool is registered{RESET}")
r = requests.get(f"{BASE}/api/status")
check("GET /api/status returns 200", r.status_code == 200)
tools = r.json().get("tools", [])
check("manage_personas in tool list", "manage_personas" in tools, f"tools: {tools}")
print()

# 1B. Ask the agent to create a persona via chat
print(f"{BOLD}Test 1B: Agent-assisted persona creation{RESET}")
info("Asking the agent to create a persona via chat...")
info("(This calls the LLM — may take 10-30 seconds)")
r = requests.post(f"{BASE}/api/chat/phase3-chat-test", json={
    "message": (
        "Create a persona with these exact details using the manage_personas tool:\n"
        "- id: news-scout\n"
        "- name: News Scout\n"
        "- description: Fetches and summarizes tech news daily\n"
        "- soul_content: You are a tech news researcher. Find and summarize the most important tech news stories. Be concise and factual.\n"
        '- cron_jobs: [{"id": "morning-news", "name": "Morning News Brief", "schedule": "daily 09:00", "prompt": "Fetch top 5 tech news stories from today and write a brief summary.", "enabled": true}]\n'
        "- tools_deny: shell\n"
        "Do not ask me for confirmation. Just create it now."
    ),
})
check("POST /api/chat returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")

if r.status_code == 200:
    response_text = r.json().get("response", "")
    info(f"Agent said: {response_text[:300]}")
    
    # Check if the persona was actually created
    time.sleep(1)  # Give filesystem a moment
    r2 = requests.get(f"{BASE}/api/personas/news-scout")
    if r2.status_code == 200:
        persona = r2.json().get("persona", r2.json())
        check("Persona 'news-scout' was created", persona.get("id") == "news-scout")
        check("Name is correct", persona.get("name") == "News Scout", f"got: {persona.get('name')}")
        check("Description is set", len(persona.get("description", "")) > 0, f"got: {persona.get('description')}")
        
        # Check SOUL.md was written
        r3 = requests.get(f"{BASE}/api/personas/news-scout/soul")
        if r3.status_code == 200:
            soul = r3.json().get("soul_content", r3.text)
            check("SOUL.md has content", len(str(soul)) > 10, f"got: {str(soul)[:100]}")
        else:
            warn(f"Could not fetch SOUL.md: {r3.status_code}")
        
        # Check cron job was created
        r4 = requests.get(f"{BASE}/api/personas/news-scout/cron")
        if r4.status_code == 200:
            cron_data = r4.json()
            jobs = cron_data.get("jobs", [])
            check("Cron job was created", len(jobs) > 0, f"got {len(jobs)} jobs")
            if jobs:
                job = jobs[0]
                check("Cron job has schedule", "09:00" in job.get("schedule", ""), f"got: {job.get('schedule')}")
                info(f"Cron job: {json.dumps(job, indent=2)[:200]}")
        else:
            warn(f"Could not fetch cron: {r4.status_code}")
    else:
        fail(f"Persona was NOT created — GET returned {r2.status_code}: {r2.text[:200]}")
        warn("The LLM may not have used the manage_personas tool. Check gateway logs.")
        errors += 1

print()

# ══════════════════════════════════════════════════════════════
# Feature 2: Cron CRUD API
# ══════════════════════════════════════════════════════════════
print(f"{BOLD}Feature 2: Cron Job CRUD API{RESET}\n")

# Create a test persona for cron testing (via API, not agent)
requests.post(f"{BASE}/api/personas", json={
    "id": "cron-test",
    "name": "Cron Test",
    "description": "Testing cron CRUD",
    "soul_content": "You are a test agent.",
})

# 2A. Add a cron job
print(f"{BOLD}Test 2A: Add cron job{RESET}")
r = requests.post(f"{BASE}/api/personas/cron-test/cron", json={
    "id": "test-job-1",
    "name": "Test Job 1",
    "schedule": "daily 10:00",
    "prompt": "Do a test thing.",
    "enabled": True,
})
check("POST cron job returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
print()

# 2B. List cron jobs
print(f"{BOLD}Test 2B: List cron jobs{RESET}")
r = requests.get(f"{BASE}/api/personas/cron-test/cron")
check("GET cron returns 200", r.status_code == 200, f"got {r.status_code}")
jobs = r.json().get("jobs", [])
check("Has 1 cron job", len(jobs) == 1, f"got {len(jobs)}: {jobs}")
if jobs:
    check("Job ID matches", jobs[0].get("id") == "test-job-1")
    check("Schedule matches", jobs[0].get("schedule") == "daily 10:00")
print()

# 2C. Add a second job
print(f"{BOLD}Test 2C: Add second cron job{RESET}")
r = requests.post(f"{BASE}/api/personas/cron-test/cron", json={
    "id": "test-job-2",
    "name": "Test Job 2",
    "schedule": "every 2h",
    "prompt": "Do another test thing.",
    "enabled": True,
})
check("POST second job returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
r = requests.get(f"{BASE}/api/personas/cron-test/cron")
jobs = r.json().get("jobs", [])
check("Now has 2 cron jobs", len(jobs) == 2, f"got {len(jobs)}")
print()

# 2D. Update a cron job
print(f"{BOLD}Test 2D: Update cron job{RESET}")
r = requests.put(f"{BASE}/api/personas/cron-test/cron/test-job-1", json={
    "schedule": "daily 11:00",
    "name": "Updated Test Job 1",
})
check("PUT cron job returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
r = requests.get(f"{BASE}/api/personas/cron-test/cron")
jobs = r.json().get("jobs", [])
job1 = next((j for j in jobs if j["id"] == "test-job-1"), None)
check("Job 1 schedule updated", job1 and job1.get("schedule") == "daily 11:00", f"got: {job1}")
check("Job 1 name updated", job1 and job1.get("name") == "Updated Test Job 1", f"got: {job1}")
print()

# 2E. Toggle a cron job
print(f"{BOLD}Test 2E: Toggle cron job{RESET}")
r = requests.post(f"{BASE}/api/personas/cron-test/cron/test-job-1/toggle")
check("POST toggle returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
r = requests.get(f"{BASE}/api/personas/cron-test/cron")
jobs = r.json().get("jobs", [])
job1 = next((j for j in jobs if j["id"] == "test-job-1"), None)
check("Job 1 is now disabled", job1 and job1.get("enabled") == False, f"got: {job1}")
print()

# 2F. Delete a cron job
print(f"{BOLD}Test 2F: Delete cron job{RESET}")
r = requests.delete(f"{BASE}/api/personas/cron-test/cron/test-job-2")
check("DELETE cron job returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
r = requests.get(f"{BASE}/api/personas/cron-test/cron")
jobs = r.json().get("jobs", [])
check("Now has 1 cron job", len(jobs) == 1, f"got {len(jobs)}: {[j['id'] for j in jobs]}")
check("Remaining job is test-job-1", jobs[0]["id"] == "test-job-1" if jobs else False)
print()

# ══════════════════════════════════════════════════════════════
# Feature 3: Model Override
# ══════════════════════════════════════════════════════════════
print(f"{BOLD}Feature 3: Model Override{RESET}\n")

# 3A. Create a persona with model override and verify it's stored
print(f"{BOLD}Test 3A: Model override stored in persona{RESET}")
r = requests.put(f"{BASE}/api/personas/cron-test", json={
    "model_override": "lmstudio/some-test-model",
})
check("PUT with model_override returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
r = requests.get(f"{BASE}/api/personas/cron-test")
persona = r.json().get("persona", r.json())
check("model_override is stored", persona.get("model_override") == "lmstudio/some-test-model",
      f"got: {persona.get('model_override')}")
info("Note: Actually testing model switching requires the override model to be loaded.")
info("The agent.py try/finally wrapper ensures the original model is always restored.")
print()

# ── Cleanup ──
print(f"{BOLD}Cleanup{RESET}")
r1 = requests.delete(f"{BASE}/api/personas/news-scout")
r2 = requests.delete(f"{BASE}/api/personas/cron-test")
requests.delete(f"{BASE}/api/sessions/phase3-chat-test")
info(f"Deleted news-scout: {r1.status_code}, cron-test: {r2.status_code}")
print()

# ── Summary ──
print(f"{BOLD}{'='*60}")
print("SUMMARY")
print(f"{'='*60}{RESET}")
if errors == 0:
    print(f"{GREEN}{BOLD}All tests passed!{RESET}")
else:
    print(f"{RED}{BOLD}{errors} test(s) failed.{RESET} Copy output into Claude Code to fix.")

sys.exit(0 if errors == 0 else 1)
