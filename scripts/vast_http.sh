#!/usr/bin/env bash
# Start (or restart) a localhost HTTP file server for tunneled downloads.
cd /root/pa3 || exit 1
pkill -f "http.server 8080" 2>/dev/null
sleep 1
setsid nohup /venv/main/bin/python -m http.server 8080 --bind 127.0.0.1 \
  > logs/http.log 2>&1 < /dev/null &
sleep 3
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/speed20.bin)
echo "server_http_code=$code"
ls -lh speed20.bin
