#!/bin/bash
# Compact 125B smoke bench: small prefill, then decode, then cached decode.
# Usage: bench2.sh <port> <label>
set -u
PORT=${1:?port}; LABEL=${2:?label}
WIKI=${WIKI:?set WIKI=/path/to/wiki.test.raw}
PROMPT=$(head -c 700 "$WIKI" \
  | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

run() { # n_predict cache_prompt tag
  curl -s "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
    -d "{\"prompt\":${PROMPT},\"n_predict\":$1,\"temperature\":0,\"cache_prompt\":$2}" \
  | python3 -c 'import json,sys
d=json.load(sys.stdin); t=d.get("timings",{})
pn=t.get("prompt_n",0); pm=t.get("prompt_ms",0); dn=t.get("predicted_n",0); dm=t.get("predicted_ms",0)
pp=1000*pn/pm if pm else 0; tg=1000*dn/dm if dm else 0
print("%s: pp=%dt %.0fms %.2f t/s | tg=%dt %.0fms %.2f t/s | %r" % (sys.argv[1], pn, pm, pp, dn, dm, tg, d.get("content","")[:40]))' "$3"
}

echo "### ${LABEL} $(date -Iseconds)"
echo -n "warmup:            "; run 8 false warm
echo -n "prefill+decode r1: "; run 128 false r1
echo -n "prefill+decode r2: "; run 128 false r2
echo -n "cached decode:     "; run 128 true cache
