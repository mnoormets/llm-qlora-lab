"""Make child-process failures visible and retain the full Colab training log."""
import subprocess
from pathlib import Path


def run_logged(command, log_path):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
    if code:
        raise RuntimeError(f'Training exited with status {code}. Actual traceback is above; full log: {path}')
    return code
