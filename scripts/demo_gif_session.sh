#!/usr/bin/env bash
# 30-second GIF demo script. Run while recording terminal with QuickTime
# (Cmd+Shift+5 → Record selected portion → save as .mov, convert to GIF
# via ffmpeg or https://ezgif.com/video-to-gif).
#
# Usage: BASE=https://alchemt-screening-tool.hf.space ./scripts/demo_gif_session.sh
#
# Plays 4 commands with deliberate pauses for readability.

set -e
BASE="${BASE:-https://alchemt-screening-tool.hf.space}"

step() {
  echo
  echo "\$ $1"
  sleep 1
  eval "$1"
  sleep 2
}

# 1. List JDs
step "curl -s $BASE/jds | jq '.[0:3] | .[] | {jd_id, title, seniority}'"

# 2. Start session
SID=$(curl -s -X POST "$BASE/screening/start" \
  -H "Content-Type: application/json" \
  -d '{"jd_id":"jd_005","max_questions":2}')
echo
echo "\$ POST /screening/start jd_005"
echo "$SID" | jq '{session_id, jd_title, question}'
SESSION=$(echo "$SID" | jq -r .session_id)
sleep 3

# 3. First answer
echo
echo "\$ POST /screening/respond  (strong candidate response)"
RESP=$(curl -s -X POST "$BASE/screening/respond" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION\",\"response\":\"I built a 3-agent LangGraph system for contract review with SqliteSaver checkpointer in dev and Postgres in prod. ~200 contracts/day, p95 12s.\"}")
echo "$RESP" | jq '{done, next_question}'
sleep 3

# 4. Final answer + report
echo
echo "\$ POST /screening/respond  (final)"
curl -s -X POST "$BASE/screening/respond" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION\",\"response\":\"For PDF parsing I use pdfplumber for digital and Unstructured for scans. Verification pass grounds every flagged clause back to its source span.\"}" \
  | jq '{done, recommendation, report_url}'
sleep 3

echo
echo "\$ GET /screening/$SESSION/report"
curl -s "$BASE/screening/$SESSION/report" | jq '{recommendation, total_cost_usd}'
echo
