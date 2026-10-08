# Paste this as ONE new Colab code cell in a free T4 runtime.
import subprocess,sys,importlib,json,shutil,os
from pathlib import Path
ROOT=Path('/content/llm-qlora-lab')
if not ROOT.exists():
    subprocess.run(['git','clone','https://github.com/mnoormets/llm-qlora-lab.git',str(ROOT)],check=True)
else:
    subprocess.run(['git','-C',str(ROOT),'pull','--ff-only'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(ROOT/'requirements-gpu.txt')],check=True)
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
importlib.invalidate_caches()
from qlora.runtime import run_logged
run_logged([sys.executable,'-u','-m','qlora.train','--steps','60','--eval-cases','16','--profile-steps','2'],ROOT/'runs/colab-profile.log')
reports=sorted((ROOT/'runs').glob('*/report.json'));report=reports[-1]
assert json.loads(report.read_text())['status']=='completed'
archive=shutil.make_archive('/content/qlora-profile-experiment','zip',root_dir=report.parent)
from google.colab import files
files.download(archive)
