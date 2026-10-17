# MIL-SIM website

Pick a controller and a map, then watch the bicycle model drive the course.

From the repository root:

```powershell
python -m autocross web
```

That opens `http://127.0.0.1:8765`. The page is served by the harness, which runs the selected controller on the selected course. The first Drive for a pair simulates the lap. Later visits replay the saved result under `runs/web_cache/`.

Opening `index.html` as a file does not work: the page needs that server.
