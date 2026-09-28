#!/usr/bin/env python3
"""Greedy determinism check against phase1/reference-greedy.jsonl. Usage: greedy-check.py <port> <out.jsonl> [tag]"""
import json, os, sys, urllib.request

port = int(sys.argv[1])
out_path = sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "run"

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference-greedy.jsonl")

refs = [json.loads(l) for l in open(ref_path)]
results = []
for r in refs:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps({"prompt": r["prompt"], "n_predict": r["n"], "temperature": 0, "cache_prompt": False, "seed": 1}).encode(),
        headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    content = d.get("content", "")
    same = content == r["content"]
    t = d.get("timings", {})
    tg = 1000 * t.get("predicted_n", 0) / max(t.get("predicted_ms", 1), 1)
    results.append({"prompt": r["prompt"], "content": content, "match": same})
    print(f"[{tag}] {'MATCH' if same else 'DIFF '} tg={tg:.2f} t/s n={t.get('predicted_n',0)} prompt={r['prompt'][:40]!r}")
    if not same:
        print(f"   ref: {r['content'][:120]!r}")
        print(f"   got: {content[:120]!r}")

with open(out_path, "a") as f:
    f.write(json.dumps({"tag": tag, "results": results}) + "\n")

print("ALL MATCH" if all(r["match"] for r in results) else "MISMATCH")
