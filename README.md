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

11 local CPU tests passed, including schema failures, split IDs, completion masks,
padding/EOS labels, GPU fail-closed preflight and an actual one-step Trainer/PEFT
interface check on a randomly initialized small model. That last check validates
API compatibility only. It is not the 7B experiment. GPU quantization, training,
adapter performance and Hub publication remain unverified until the Colab run.

Sources: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization),
[Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer),
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).


Notebook format and top-to-bottom CPU contract execution were verified with a
fresh Jupyter kernel; all six code cells completed without errors. GPU cells were
explicitly skipped. See notebook-validation.json; this does not validate a GPU run.
