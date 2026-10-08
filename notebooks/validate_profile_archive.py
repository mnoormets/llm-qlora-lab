"""Inspect user-exported JSON/safetensors evidence without loading pickle artifacts."""
import argparse, collections, hashlib, io, json, math, re, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlora.data import load_rows, score_predictions


def trace_events(stream):
    decoder = json.JSONDecoder()
    buffer = ""
    while True:
        chunk = stream.read(65536)
        if not chunk:
            raise ValueError("Missing traceEvents array")
        buffer += chunk
        match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
        if match:
            buffer = buffer[match.end():]
            break
        if len(buffer) > 1048576:
            raise ValueError("Unexpected trace header")
    while True:
        buffer = buffer.lstrip(" \r\n\t,")
        if buffer.startswith("]"):
            return
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError("Truncated trace")
            buffer += chunk
            if len(buffer) > 16777216:
                raise ValueError("Oversized trace event")
            continue
        if not isinstance(value, dict):
            raise ValueError("Expected trace object")
        yield value
        buffer = buffer[end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    archive = Path(args.archive)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    imported = ROOT / "runs" / ("profile-imported-" + archive_hash[:12])
    imported.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if member.filename.startswith(("/", "\\")) or ".." in parts:
                raise ValueError("Unsafe member")
        allowed = ["report.json", "memory-events.jsonl", "profile/operators.txt", "profile/training-trace.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors", "checkpoints/checkpoint-60/trainer_state.json"]
        for name in allowed:
            out = imported / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1048576)
        report = json.loads(z.read("report.json"))
        assert report["status"] == "completed" and report["steps"] == 60
        for name, expected in report["adapter_files"].items():
            assert hashlib.sha256(z.read("adapter/" + name)).hexdigest() == expected
    for split, expected in report["dataset_sha256"].items():
        assert hashlib.sha256((ROOT / "fixtures" / (split + ".jsonl")).read_bytes()).hexdigest() == expected
    state = json.loads((imported / "checkpoints/checkpoint-60/trainer_state.json").read_text())
    assert state["global_step"] == 60
    from safetensors.torch import load_file
    tensors = load_file(str(imported / "adapter/adapter_model.safetensors"))
    assert all(t.isfinite().all().item() for t in tensors.values())
    parameters = sum(t.numel() for t in tensors.values())
    assert parameters == report["trainable_parameters"]
    rows = load_rows(ROOT / "fixtures/test.jsonl")[:16]
    scores = {}
    for key in ["baseline", "adapted"]:
        details = report[key]["details"]
        assert [x["id"] for x in details] == [x["id"] for x in rows]
        assert [x["expected"] for x in details] == [x["expected"] for x in rows]
        score = score_predictions(rows, [x["prediction"] for x in details])
        scores[key] = {k: v for k, v in score.items() if k != "details"}
        for k, v in scores[key].items():
            assert report[key][k] == v
    diagnostic = score_predictions(rows, [re.sub(r'^```(?:json)?\s*|\s*```$', '', x["prediction"]) for x in report["baseline"]["details"]])
    memory = [json.loads(line) for line in (imported / "memory-events.jsonl").read_text().splitlines()]
    assert all(x["cuda_memory_available"] for x in memory)
    for row in memory:
        assert 0 <= row["allocated_bytes"] <= row["reserved_bytes"]
        assert row["interval_peak_allocated_bytes"] >= row["allocated_bytes"]
        assert row["interval_peak_reserved_bytes"] >= row["reserved_bytes"]
    allocated = max(x["interval_peak_allocated_bytes"] for x in memory)
    reserved = max(x["interval_peak_reserved_bytes"] for x in memory)
    assert math.isclose(allocated / 2**30, report["peak_allocated_gpu_gib"])
    assert math.isclose(reserved / 2**30, report["peak_reserved_gpu_gib"])
    ends = [x for x in memory if x["phase"] == "optimizer_interval_end"]
    assert [x["step"] for x in ends] == list(range(1, 61))
    categories = collections.Counter()
    durations = collections.Counter()
    event_count = 0
    profiler_steps = []
    with (imported / "profile/training-trace.json").open(encoding="utf-8") as stream:
        for event in trace_events(stream):
            event_count += 1
            category = event.get("cat", "uncategorized")
            categories[category] += 1
            if event.get("ph") == "X":
                durations[category] += max(0, float(event.get("dur", 0)))
            if event.get("name", "").startswith("ProfilerStep#"):
                profiler_steps.append(event["name"])
    assert categories["kernel"] > 0 and categories["cpu_op"] > 0
    result = {
        "archive_sha256": archive_hash, "source_commit": report["source_commit"],
        "artifact_hashes_verified": True, "dataset_hashes_verified": True,
        "checkpoint_global_step": 60, "adapter_tensor_count": len(tensors),
        "adapter_parameters": parameters, "adapter_tensors_finite": True,
        "scores_recomputed": scores,
        "baseline_fence_stripping_diagnostic": {k: v for k, v in diagnostic.items() if k != "details"},
        "memory": {"event_count": len(memory), "phase_counts": dict(collections.Counter(x["phase"] for x in memory)),
                   "all_60_optimizer_steps_present": True, "peak_allocated_gib": allocated/2**30,
                   "peak_reserved_gib": reserved/2**30,
                   "scope": "PyTorch allocator interval peaks, not total device VRAM"},
        "profile": {"file_bytes": (imported / "profile/training-trace.json").stat().st_size,
                    "trace_events": event_count, "category_counts": dict(categories),
                    "summed_event_duration_ms": {k: v/1000 for k, v in durations.items()},
                    "profiler_step_names": profiler_steps,
                    "scope": "Two optimizer intervals with profiler overhead; overlapping event durations are not wall-clock shares"},
        "limitations": ["16 synthetic test cases; no independent real-document validation", "No local GPU inference replay", "No multi-GPU, AWQ or serving benchmark demonstrated", "Profiler adds overhead; no controlled speedup comparison"]}
    results = ROOT / "results/profile-t4"
    results.mkdir(parents=True, exist_ok=True)
    for name in ["report.json", "memory-events.jsonl", "profile/operators.txt"]:
        target = results / Path(name).name
        shutil.copyfile(imported / name, target)
    (results / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
