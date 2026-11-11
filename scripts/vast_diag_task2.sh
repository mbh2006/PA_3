#!/usr/bin/env bash
# Diagnose the Task 2 data build: what was downloaded and why the zip read failed.
cd /root/pa3 || exit 1
echo "=== annotation files ==="
ls -lh task2_data/annotations/ 2>/dev/null || echo "no annotations dir"
echo "=== zip members ==="
/venv/main/bin/python - <<'PY'
import zipfile, pathlib
p = pathlib.Path('/root/pa3/task2_data/annotations/annotations_trainval2014.zip')
if not p.exists():
    print("zip missing")
else:
    print("size", p.stat().st_size)
    try:
        z = zipfile.ZipFile(p)
        names = z.namelist()
        print("members", len(names))
        for n in names[:8]:
            print("  ", n)
    except Exception as exc:
        print("bad zip:", exc)
PY
echo "=== build log head ==="
head -20 logs/task2_build.log
