#!/usr/bin/env python3
"""Run a decode request against llama-server and sample GPU busy% in parallel."""
import json, sys, threading, time, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
n_predict = int(sys.argv[2]) if len(sys.argv) > 2 else 256
tag = sys.argv[3] if len(sys.argv) > 3 else "run"

import os
wiki = os.environ.get("WIKI")
prompt = open(wiki).read(700) if wiki else (sys.argv[4] if len(sys.argv) > 4 else "")

busy = {"card0": [], "card1": []}
stop = False

def sample():
    while not stop:
        for c in busy:
            try:
                busy[c].append(int(open(f"/sys/class/drm/{c}/device/gpu_busy_percent").read()))
            except Exception:
                pass
        time.sleep(0.1)

th = threading.Thread(target=sample, daemon=True)
th.start()

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": False}).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=600))
wall = time.time() - t0
stop = True
th.join(timeout=1)

t = d.get("timings", {})
print(f"[{tag}] pp={t.get('prompt_n',0)}t {1000*t.get('prompt_n',0)/max(t.get('prompt_ms',1),1):.0f} t/s | "
      f"tg={t.get('predicted_n',0)}t {1000*t.get('predicted_n',0)/max(t.get('predicted_ms',1),1):.2f} t/s | wall {wall:.1f}s")
for c, v in busy.items():
    if v:
        print(f"  {c}: busy mean {sum(v)/len(v):.1f}%  max {max(v)}%  samples {len(v)}")
