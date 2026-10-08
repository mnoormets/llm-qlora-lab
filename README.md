# 7B QLoRA Extraction Lab

A bounded, reproducible GPU experiment for Estonian invoice-to-JSON extraction.
Pinned Qwen2.5-7B-Instruct, NF4 4-bit weights, PEFT LoRA and completion-only
supervision. **The GPU run has not been executed yet.** CPU contract checks are
not a fine-tuning result and no adapter has been uploaded to Hugging Face.

## Run on a free Colab GPU

[Open the notebook](https://colab.research.google.com/github/mnoormets/llm-qlora-lab/blob/main/notebooks/colab_qlora.ipynb).
Choose Runtime -> Change runtime type -> T4 GPU (if available), then Run all.
The notebook clones this repository, installs pinned GPU libraries, checks CUDA,
checks the dataset and runs baseline generation before training. Colab availability
is controlled by Google; the notebook does not rent a GPU or call paid inference.

The default first run is 60 optimizer steps, batch size 1, accumulation 8,
sequence limit 768 and 16 final-test examples. This is an initial experiment,
not sufficient evidence of general financial-document quality.

## Data and evaluation

480 authored synthetic invoices: 400 training, 40 validation and 40 test.
Entity IDs are disjoint. Training uses wording templates 0-2; validation uses 3,
final test uses 4-5. Missing fields and untrusted instruction text are included.
Templates are still artificial: do not describe results as real-contract accuracy.
Dataset checksums are recorded. `python -m qlora.data` reproduces the fixtures.

The same fixed test subset is scored before and after adaptation: JSON validity,
exact match, per-field accuracy, raw predictions and failures. Unknown fields,
duplicate JSON keys, malformed amounts/dates and prose around JSON fail validation.
The test set is not used for gradient updates or checkpoint selection.

## Training engineering

- Pinned model revision; safetensors and no remote custom code.
- CUDA preflight before any model download; minimum 14 GiB GPU memory.
- fp16 on hardware without bf16 support; bf16 only when supported.
- NF4 and double quantization; rank 16/alpha 32 attention adapters.
- Gradient checkpointing, paged 8-bit optimizer and cosine learning-rate schedule.
- Only LoRA parameters are trainable; prompt and padding labels are masked.
- Overlong examples fail rather than silently truncate the supervised answer.
- Validation loss and checkpoints every 20 steps; maximum two saved checkpoints.
- Actual training history, GPU peak allocated/reserved memory, runtime versions,
  dataset hashes and adapter-file hashes are saved under ignored runs/.
- CUDA OOM writes a failed receipt with recovery suggestions, not a success metric.

Run `python -m qlora.train --preflight-only`, then
`python -m qlora.train --steps 60 --eval-cases 16` in the GPU environment.
Use the notebook's final ZIP download to retain artifacts before Colab disconnects.
No W&B account, Hugging Face token or paid service is required for the initial run.
Optional Hub upload is disabled by default and uses an interactively supplied token,
never a token embedded in the notebook or committed to Git.

## Validation so far

17 local CPU tests passed, including schema failures, split IDs, completion masks,
padding/EOS labels, GPU fail-closed preflight and an actual one-step Trainer/PEFT
interface check on a randomly initialized small model. That last check validates
API compatibility only. It is not the 7B experiment. The imported Colab GPU run is now artifact-verified as described below. Hub
publication and an independent inference reload remain unverified.

Sources: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization),
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer),
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).


Notebook format and top-to-bottom CPU contract execution were verified with a
fresh Jupyter kernel; all six code cells completed without errors. GPU cells were
explicitly skipped. See notebook-validation.json; this does not validate a GPU run.


Dataset files use explicit LF line endings so committed-file hashes match on
Windows and Linux. This fixes the first manifest's platform-dependent line-ending
hashes; document text and labels are unchanged. Training receipts hash the actual
input files independently, including runs started from the earlier commit.

## Colab failure visibility and T4 precision
Native BF16 support is checked with `including_emulation=False`; T4 uses FP16. Colab streams the child-process traceback and saves `runs/colab-training.log`. Failed receipts include the exception message. A generic CalledProcessError alone does not identify the failure cause. These changes were CPU-contract tested; the subsequent GPU retry completed; imported artifacts were inspected.

## Transformers 5 training arguments
The first GPU retry loaded the model but failed on removed `warmup_ratio`. Production now uses an explicit integer `warmup_steps` (3 of 60 steps) and constructs TrainingArguments before loading weights. Tests bind every production setting to the pinned Transformers signature and the real CPU Trainer smoke test uses the same options with CPU-only hardware overrides. These tests verify API compatibility. The separately imported GPU run is described below.

## Completed Colab GPU experiment

Actual exported receipt: [results/colab-report.json](results/colab-report.json).
Artifact inspection and independently recomputed saved-prediction metrics:
[results/artifact-validation.json](results/artifact-validation.json).

- Qwen2.5-7B-Instruct, pinned revision; NF4 QLoRA on Tesla T4, FP16.
- 60 optimizer steps, 400 authored training invoices; 10,092,544 LoRA parameters.
- Training runtime: 528.3 seconds. Full recorded experiment: 732.7 seconds.
- Peak allocated GPU memory: 8.773 GiB; peak reserved: 9.520 GiB.
- Adapter: 16/16 exact schema-compliant outputs on the fixed synthetic test subset.
- Strict baseline: 0/16 schema-compliant exact outputs. It generally returned
  Markdown-wrapped JSON, which the strict production contract rejects.
- Diagnostic after stripping only outer Markdown fences: baseline 13/16 exact,
  15/16 schema-valid and 90.625% field accuracy. Adapter: 16/16 exact and valid.
  This diagnostic was computed after the experiment and did not change training.

This demonstrates a functioning GPU fine-tuning pipeline and improvement on a
small synthetic extraction/format-following task. It is not 100% real-document
accuracy, broad superiority over the base model, or evidence of production scale.
The validator checked dataset hashes, all exported adapter-file hashes, checkpoint
step 60 and 224 finite nonzero LoRA tensors. It did not replay GPU inference locally.
The exported report lacks a source-commit field, so exact code provenance cannot
be inferred from it; model revision and dataset hashes are present. Model weights,
optimizer state and pickle checkpoints are intentionally not committed to Git.

## Memory tracing and optional training profiler

MemoryTrace records allocator allocated/reserved and interval peaks at optimizer begin/end, accumulated microbatch end and before/after optimizer. End-of-step snapshots alone cannot locate every forward/backward peak; the bounded profiler adds operator-level evidence. Allocator bytes exclude CUDA context, some external allocations and other processes; do not equate reserved bytes to used tensor bytes.

```
python -m qlora.train --steps 60 --eval-cases 16 --profile-steps 2
```

This saves memory-events.jsonl plus profile/training-trace.json and operator tables. Profiling is off by default, bounded to five optimizer intervals with one warmup; it adds overhead. Report peak is the maximum recorded across intervals and training evaluation, not the last interval after reset. Existing completed GPU results predate instrumentation and are unchanged. New GPU telemetry still requires an actual Colab run; CPU Trainer/profile integration is tested.
