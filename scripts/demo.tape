# vhs tape — Screening Tool API demo
# Render with: vhs scripts/demo.tape  →  assets/demo.gif

Output assets/demo.gif

Set FontSize 14
Set Width 1200
Set Height 720
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set PlaybackSpeed 1.0
Set Padding 20

Hide
Type "clear"
Enter
Type "BASE=https://alchemt-screening-tool.hf.space"
Enter
Show

Sleep 500ms
Type "# 1. List available job descriptions"
Sleep 500ms
Enter
Type "curl -s $BASE/jds | jq '.[0:3] | .[] | {jd_id, title, seniority}'"
Enter
Sleep 4s

Type "# 2. Start a screening session"
Sleep 500ms
Enter
Type `RES=$(curl -s -X POST $BASE/screening/start -H 'Content-Type: application/json' -d '{"jd_id":"jd_005","max_questions":2}')`
Enter
Type "echo $RES | jq '{jd_title, question}'"
Enter
Sleep 4s
Type "SID=$(echo $RES | jq -r .session_id)"
Enter
Sleep 1s

Type "# 3. Submit a strong candidate response"
Sleep 500ms
Enter
Type `curl -s -X POST $BASE/screening/respond -H 'Content-Type: application/json' -d "{\"session_id\":\"$SID\",\"response\":\"I built a 3-agent LangGraph system for contract review with SqliteSaver in dev and Postgres in prod, ~200 contracts/day, p95 12s.\"}" | jq '{done, next_question}'`
Enter
Sleep 5s

Type "# 4. Final answer + recommendation"
Sleep 500ms
Enter
Type `curl -s -X POST $BASE/screening/respond -H 'Content-Type: application/json' -d "{\"session_id\":\"$SID\",\"response\":\"For PDF parsing I use pdfplumber for digital and Unstructured for scans. A verification pass grounds every flagged clause back to its source span.\"}" | jq '{done, recommendation}'`
Enter
Sleep 5s

Type "# 5. Fetch the full report"
Sleep 500ms
Enter
Type "curl -s $BASE/screening/$SID/report | jq '{recommendation, total_cost_usd}'"
Enter
Sleep 4s
