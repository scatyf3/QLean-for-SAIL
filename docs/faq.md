# FAQ

## How do I set my model_name?

Run `qlean --list-models` or `qlean --list-models MODEL_NAME`. The name must match the official HuggingFace model stub. Unlisted models fall back to an architecture-level preset or a default recipe; a custom `--recipe` is recommended. See [manual.md](manual.md).

## Input-format auto-detection failed?

Check that the model's `config.json`, `model.safetensors.index.json`, and weight files are all present, then override explicitly once you know the source format:

```bash
qlean --model_name <name> \
      --model_path /path/to/model \
      --save_path /path/to/output \
      --input-format FP8 \
      --output-format INT8_W8A8
```

`--input-format` accepts only `BF16`, `FP8`, and `NVFP4`. For a checkpoint that mixes source formats, use a recipe and set `src` per region.

## Accuracy dropped after conversion?

- If in auto mode: Did a built-in recipe apply at all? `--model_name` must match a name from `qlean --list-models`, otherwise an architecture-level or generated config is used.
- If in manual mode: Does the recipe describe the checkpoint? Check `src`, the tensor layout, and the scale settings; keep sensitive layers unquantized with a copy region; confirm `dequant_dtype` is fp32.

Three layers of evidence, from cheapest to decisive: `--verify bound` reports (per-tensor error against the theoretical bound), a standalone numerical comparison against a reference checkpoint, and end-to-end benchmark evaluation. Only the last one settles an accuracy claim.

## `--verify bound` produced no `qlean_verify/` directory?

Verification only runs inside `GeneralQuantEngineModifier`. If the matched recipe uses a legacy modifier, the CLI logs one warning and carries on without it — nothing fails, nothing is written:

    --verify bound was given, but the recipe has no GeneralQuantEngineModifier; ignoring

Grep the run log for `[verify]`:

- the line above → point `--recipe` at a recipe with `GeneralQuantEngineModifier`
- `[verify] Bound checks enabled; details written to <path>` → it is active; `<path>` is authoritative, and the directory only appears once the first shard is written
- no `[verify]` line at all → the installed build predates the feature; confirm with `qlean --help | grep -- --verify`

Other branches: [verification.md](verification.md#troubleshooting).

## What is `qlean_config.json`?

It records the tool version, timestamps, model paths, input/output formats, the recipe path and its full content, the modifier chain, and the CLI arguments. Use it to trace what ran, to reproduce a conversion elsewhere, or to attach it when reporting an issue. With `--recipe`, `input_format` and `output_format` are recorded as `user_specified`; `recipe_content` describes what actually ran.
