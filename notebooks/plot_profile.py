"""Plot validated local T4 allocator/loss evidence; optional dependency: matplotlib."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parents[1]/"results/profile-t4"
rows=[json.loads(x) for x in (out/"memory-events.jsonl").read_text().splitlines()]
ends=[r for r in rows if r["phase"]=="optimizer_interval_end"]
fig,axes=plt.subplots(2,1,figsize=(9,7),constrained_layout=True)
for key,label in [("allocated_bytes","Allocated at step end"),("reserved_bytes","Reserved at step end"),("interval_peak_allocated_bytes","Interval peak allocated")]:
    axes[0].plot([r["step"] for r in ends],[r[key]/2**30 for r in ends],label=label)
axes[0].set(xlabel="Optimizer step",ylabel="PyTorch allocator memory (GiB)",title="Qwen2.5-7B QLoRA · Tesla T4 · 60 steps")
axes[0].legend();axes[0].grid(alpha=.2)
report=json.loads((out/"report.json").read_text())
train=[r for r in report["training_history"] if "loss" in r]
evaluation=[r for r in report["training_history"] if "eval_loss" in r]
axes[1].semilogy([r["step"] for r in train],[r["loss"] for r in train],"o-",label="Logged training loss")
axes[1].semilogy([r["step"] for r in evaluation],[r["eval_loss"] for r in evaluation],"s-",label="Validation loss")
axes[1].set(xlabel="Optimizer step",ylabel="Loss (log scale)",title="Synthetic invoice task; profiling overhead included")
axes[1].legend();axes[1].grid(alpha=.2)
fig.savefig(out/"memory-and-loss.png",dpi=150)
plt.close(fig)
