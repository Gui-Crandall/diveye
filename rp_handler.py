import os, json, subprocess, tempfile, pathlib, shlex
import runpod

# Helper: if user passes a URL, we download; if path, we use it directly.
def _materialize_csv(src: str, workdir: str, filename: str) -> str:
    p = pathlib.Path(workdir) / filename
    if src.startswith("http://") or src.startswith("https://"):
        import urllib.request
        urllib.request.urlretrieve(src, p.as_posix())
        return p.as_posix()
    # allow mounted volume paths or relative paths in image
    src_p = pathlib.Path(src)
    if src_p.exists():
        return src_p.as_posix()
    raise FileNotFoundError(f"CSV not found or bad URL: {src}")

def _run(cmd_list, cwd=None):
    proc = subprocess.run(cmd_list, cwd=cwd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr

def handler(event):
    """
    Expected event['input']:
      {
        "model": "distilroberta-base",        # any HF model name (required)
        "train_csv": "<URL or path>",         # required
        "test_csv": "<URL or path>",          # required
        "extra_args": "--foo bar"             # optional passthrough to diveye.py
      }
    Returns:
      { "status": "ok", "stdout": "...", "stderr": "..."}  OR  {"status":"error", ...}
    """
    inp = event.get("input", {})
    model = inp.get("model")
    train_src = inp.get("train_csv")
    test_src  = inp.get("test_csv")
    extra = inp.get("extra_args", "")

    if not model or not train_src or not test_src:
        return {"status": "error", "message": "Required fields: model, train_csv, test_csv"}

    with tempfile.TemporaryDirectory() as td:
        train_csv = _materialize_csv(train_src, td, "train.csv")
        test_csv  = _materialize_csv(test_src, td, "test.csv")

        # Build command to call the original CLI
        cmd = [
            "python3", "diveye.py",
            f"--model={model}",
            f"--train_dataset={train_csv}",
            f"--test_dataset={test_csv}"
        ]

        if extra:
            # naive split; keep simple flags space-separated
            cmd.extend(shlex.split(extra))

        code, out, err = _run(cmd, cwd="/workspace")

        result = {
            "status": "ok" if code == 0 else "error",
            "exit_code": code,
            "stdout": out[-30000:],   # trim huge logs
            "stderr": err[-30000:]
        }
        return result

runpod.serverless.start({"handler": handler})
