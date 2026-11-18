# CLI Reference

```bash
qlean --model_name MODEL --model_path DIR --save_path DIR [options]
```

## Required Arguments

| Argument | Description | Example |
|:---|:---|:---|
| `--model_name` | Model identifier for built-in recipe matching. Use the official HuggingFace model stub. | `Qwen/Qwen3-235B-A22B` |
| `--model_path` | Local path to the source model directory | `/path/to/Qwen3-235B-A22B` |
| `--save_path` | Output directory | `/path/to/Qwen3-235B-A22B-INT8` |

## Optional Arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--output-format` | `INT8_W8A8` | Target output format |
| `--input-format` | Auto-detect | Source format: `BF16` / `FP8` / `NVFP4` |
| `--recipe` | Auto-select | Path to a custom Recipe YAML |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--verify` | follow Recipe | `off` / `bound`. Overrides `verify` in the Recipe; applies only to `GeneralQuantEngineModifier`. See [verification.md](verification.md). |

## Helper Commands

| Command | Purpose |
|:---|:---|
| `qlean --list-formats` | List all supported conversion paths and aliases |
| `qlean --list-models` | List all models with built-in recipes |
| `qlean --list-models MODEL_NAME` | Check recipe support for a specific model |
| `qlean --examples` | Show usage examples |
| `qlean -h` | Show full help |

## Format Names

The source format is auto-detected from the model files and rarely needs to be set manually. `--input-format` accepts only `BF16`, `FP8`, and `NVFP4`; `FP8` means an FP8 block-wise checkpoint.

Target names for `--output-format` and their accepted aliases. All names are case-insensitive.

| Format | Aliases |
|:---|:---|
| INT8_W8A8 | W8A8, INT8 |
| FP8_BLOCK | FP8-BLOCK |
| FP8_DYNAMIC | FP8-DYNAMIC, FP8-CHANNEL, FP8_CHANNEL |
| MXFP4 | — |
| BF16 | — |

## Output Directory Rules

`--save_path` must not be the same as, or nested inside/around, `--model_path`. Repeated runs resume by default; any config change alters the fingerprint and reprocesses the affected shards.
