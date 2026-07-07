Output docs/demo.mp4
Require docker

Set FontSize 18
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 26
Set TypingSpeed 35ms

# --- silent setup, hidden from the recording ---
Hide
Type `cd /Users/bisman.singh/test/frugal-router` Enter
Type `mkdir -p /tmp/in /tmp/out` Enter
Type `printf '[{"task_id":"a","prompt":"What is 15 percent of 240?"},{"task_id":"b","prompt":"Sentiment? Justify in one sentence: I loved the crisp display but the battery dies by noon."},{"task_id":"c","prompt":"Write a Python function named add(a, b) that returns their sum."}]' > /tmp/in/tasks.json` Enter
Type `clear` Enter
Show

# --- the real run, on the public submission image ---
Sleep 700ms
Type `docker run --rm -e FIREWORKS_API_KEY -e ALLOWED_MODELS \` Enter
Type `  -v /tmp/in:/input:ro -v /tmp/out:/output \` Enter
Type `  bismansinghmadaan/frugal-router:v1`
Sleep 800ms
Enter
Sleep 9s

Type `cat /tmp/out/results.json | python3 -m json.tool`
Sleep 600ms
Enter
Sleep 6s
