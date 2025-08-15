import os, json, subprocess, tempfile, pathlib, shlex, base64, csv
import runpod

def _write_b64(b64_str, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_str))
    return path

def _download(url, path):
    import urllib.request
    urllib.request.urlretrieve(url, path)
    return path

def _materialize(inp, key_url, key_b64, default_name, workdir):
    p = pathlib.Path(workdir) / default_name
    if inp.get(key_b64):
        return _write_b64(inp[key_b64], p.as_posix())
    if inp.get(key_url):
        return _download(inp[key_url], p.as_posix())
    return None

def handler(event):
    inp = event.get("input", {})
    model = inp.get("model")
    extra = inp.get("extra_args", "")

    if not model:
        return {"status": "error", "message": "model is required"}

    with tempfile.TemporaryDirectory() as td:
        # Accept base64 or URL for train/test CSVs
        train_csv = _materialize(inp, "train_csv", "train_csv_b64", "train.csv", td)
        test_csv  = _materialize(inp, "test_csv",  "test_csv_b64",  "test.csv",  td)

        # Convenience: if user passed raw text, make a one-row test.csv
        if not test_csv and inp.get("test_text"):
            test_csv = os.path.join(td, "test.csv")
            with open(test_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["text"])
                w.writerow([inp["test_text"]])

        if not train_csv or not test_csv:
            return {"status": "error",
                    "message": "Need at least train_csv(_b64) and test_csv(_b64) or test_text."}

        cmd = ["python3", "diveye.py",
               f"--model={model}",
               f"--train_dataset={train_csv}",
               f"--test_dataset={test_csv}"]
        if extra:
            cmd.extend(shlex.split(extra))

        proc = subprocess.run(cmd, cwd="/workspace", text=True, capture_output=True)
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-30000:],
            "stderr": proc.stderr[-30000:]
        }

runpod.serverless.start({"handler": handler})
