# Manual Mode

Manual mode means passing your own recipe with `--recipe path/to/recipe.yaml`.
Your file becomes the single source of truth; it overrides auto mode entirely:

- `--output-format` and the built-in recipes are ignored.
- `--model_name` is still written to `qlean_config.json` for the record, but it no longer selects a recipe.
- Every decision — which tensors, which format, how — comes from the regions in your recipe.

Use it when the model has no built-in recipe, when the checkpoint mixes source formats, or when you want a layer split the built-ins do not produce.

## Recipe

`GeneralQuantEngineModifier` is the modifier this page describes. The two
lines above it are boilerplate you copy unchanged; **in practice you only edit
the fields nested under `GeneralQuantEngineModifier`**.

```yaml
quant_stage:
  quantization_modifiers:
    GeneralQuantEngineModifier:
      # Output Format
      primary_scheme: W8A8              # Optional (default: null); required by compressed_tensors.
      config_writer: compressed_tensors # Optional; compressed_tensors | none.
      sglang_wa: auto                   # Optional; auto | true | false. auto enables WA for MXFP4.
      scale_naming: underscore          # Optional; underscore: x.weight_scale | dot: x.scale.

      # Execution
      num_gpus: 0                       # Optional; 0 = all visible GPUs.
      resume: true                      # Optional; reuse completed shards with matching fingerprints.
      force_rerun: false                # Optional; skip cleanup confirmation, not resume checks.

      # Verification
      verify: "off"                     # Optional; "off" | bound. bound writes CSV reports only.

      # Source dequantization
      dequant_dtype: fp32               # Optional; fp32 | bf16. Intermediate precision.
      dequant_block_size: 128           # Optional; FP8 block source block size.
      int4_group_size: 32               # Optional; INT4 source group size; positive integer.
      nvfp4_group_size: 16              # Optional; NVFP4 source group size.

      # Regions: first match wins; every pattern must match source tensors.
      # src: bf16 | fp8_block | nvfp4 | int4 | mxfp4 | mxfp8 (case-insensitive).
      # dst: W8A8 | MXFP4 | FP8_DYNAMIC | FP8_BLOCK | bf16 | passthrough.
      regions:                          # Required; nonempty ordered list.
        - name: ignore                  # Copy unquantized tensors; omit src/dst.
          match:                        # Optional (default: "*"); full-name glob or OR list; no re:.
            - "*norm*"
            - "*embed_tokens.weight"
            - "*lm_head.weight"

        - name: experts                 # Required; unique name. experts updates expert_dtype.
          match: "*.experts.*_proj.weight"
          src: bf16                     # Optional; source format. bf16 means no dequantization.
          dst: W8A8                     # Optional (default: null/copy); required for non-bf16 src.
          when: always                  # Optional; always | has_source_scale | no_source_scale.

        # Required last region: copy only, no src/dst. Zero routed hits are allowed.
        - name: fallback
          match: "*"
```

## Modifier Fields

### Output Format

| Field | Default | Notes |
|:---|:---|:---|
| `primary_scheme` | — | The scheme recorded in the output `config.json`: `W8A8` \| `FP8_BLOCK` \| `FP8_DYNAMIC` \| `MXFP4`. Each region's own `dst` is what actually quantizes a tensor; `primary_scheme` is the global label, required when `config_writer: compressed_tensors` |
| `config_writer` | `compressed_tensors` | `compressed_tensors` writes standard `quantization_config`; `none` skips this writer |
| `sglang_wa` | `auto` | Inject an sglang weight/activation block into the config. `auto` enables it only for MXFP4; override with `true` / `false` |
| `scale_naming` | `underscore` | Output scale key style: `underscore` → `x.weight_scale`, `dot` → `x.scale`. Match whatever the target runtime expects |

### Execution

| Field | Default | Notes |
|:---|:---|:---|
| `num_gpus` | `0` | `0` = all visible GPUs; a positive integer caps the worker count |
| `resume` | `true` | Reuse shards whose fingerprint matches and whose output is complete; set `false` for a full rerun |
| `force_rerun` | `false` | Skip the confirmation prompt when clearing existing output after a fingerprint mismatch |

### Verification

| Field | Default | Notes |
|:---|:---|:---|
| `verify` | `"off"` | `"off"` \| `bound`. `bound` checks supported quantization operations against theoretical error bounds. Quote the value: bare `off` is a YAML boolean. See [verification.md](verification.md) |

### Source Dequantization

| Field | Default | Notes |
|:---|:---|:---|
| `dequant_dtype` | `fp32` | Intermediate precision after dequantization; `fp32` \| `bf16` |
| `dequant_block_size` | `128` | Block size of an `fp8_block` source |
| `int4_group_size` | `32` | Group size of an `int4` source |
| `nvfp4_group_size` | `16` | Group size of an `nvfp4` source |

### Regions

`regions` is a required, non-empty, ordered list. For each tensor the **first** region whose `match` hits wins, so order matters: put narrow patterns before broad ones, and keep the catch-all last.

| Field | Default | Notes |
|:---|:---|:---|
| `name` | required | Unique across regions; appears in the `[region hits]` report |
| `match` | `*` | Full tensor-name glob, or a list of globs (OR). `*` spans dots. Regex is rejected |
| `src` | `bf16` | Source format: `bf16` \| `fp8_block` \| `nvfp4` \| `int4` \| `mxfp4` \| `mxfp8`, case-insensitive |
| `dst` | copy | Target format: `W8A8` \| `FP8_BLOCK` \| `FP8_DYNAMIC` \| `MXFP4` \| `bf16` \| `passthrough` (see the rules below for what omitting it means) |
| `when` | `always` | `always` \| `has_source_scale` \| `no_source_scale`. Narrows the region to tensors whose source does / does not carry scale metadata — e.g. `when: has_source_scale` lets one region act on FP8 tensors (which ship a scale key) |

Beyond these, a region supports a few advanced reshape knobs (`transpose`, `split_gate_up`, `expert_mode`, `chunked`); see [`examples/EngineFullReference.yaml`](../examples/EngineFullReference.yaml).

Rules the engine enforces:

- The last region must be the catch-all `match: "*"` with `when: always`; it copies whatever nothing else matched.
- Every non-fallback region must match at least one tensor, otherwise the run aborts.
- Omitting `dst` copies the tensor unchanged, which is valid **only for a `bf16` source** (a quantized source with no `dst` raises "missing dst for non-bf16 source"). To emit BF16 from a quantized source, set `dst: bf16`.
- `dst: passthrough` keeps the packed bytes and the source scale instead of dequant→requant; valid only for `bf16`, `fp8_block`, `mxfp4`, `mxfp8` sources, and it cannot be combined with `transpose: true`, `split_gate_up`, or `expert_mode: unpack`.

## Basic Recipe

BF16 2D projection weights to W8A8. Written for a Llama-style checkpoint; adjust `match` to the tensor names your model actually uses.

```yaml
quant_stage:
  quantization_modifiers:
    GeneralQuantEngineModifier:
      primary_scheme: W8A8       # Required; global quantization scheme
      regions:                   # Required; ordered, first match wins
        # 1. Quantize the projection layers to W8A8
        - name: linear           # Required; unique region name
          match: "*_proj.weight" # Optional (default "*"); this pattern selects the 2D projection weights
          src: bf16              # Optional (default bf16); source format
          dst: W8A8              # Required to quantize; omitting it copies the tensor

        # 2. Fallback: everything unmatched is copied
        - name: fallback         # Omit src/dst
          match: "*"             # Matches all remaining tensors unconditionally
```

```bash
qlean --model_name Qwen/Qwen3-8B \
      --model_path /path/to/Qwen3-8B \
      --save_path /path/to/Qwen3-8B-mixed \
      --recipe /path/to/recipe.yaml
```

## Mixed-Precision Recipe

Giving different regions different targets produces a mixed-precision model in a single pass. GLM-5.3-Flash (FP8 block-wise source): MoE experts to MXFP4, Dense layers to FP8_DYNAMIC, norms and embeddings copied untouched.

```yaml
quant_stage:
  quantization_modifiers:
    GeneralQuantEngineModifier:
      primary_scheme: MXFP4             # Required; global quantization scheme
      dequant_block_size: 128           # Block size of the FP8 block-wise source
      dequant_dtype: fp32               # Precision of the dequantization math
      regions:
        # 1. Sensitive layers: copied, never quantized
        - name: ignore                  # Omit src/dst
          match:                        # Full tensor-name glob, single value or list
            - "*norm*"
            - "*embed_tokens.weight"
            - "*lm_head.weight"
        # 2. MoE experts -> MXFP4
        - name: experts
          match: "*.experts.*_proj.weight"
          src: fp8_block                # Source is stored FP8 block-wise
          when: has_source_scale        # Match only tensors whose source carries scale metadata
          dst: MXFP4
        # 3. Dense layers -> FP8_DYNAMIC
        - name: dense
          match: "*.weight"
          src: fp8_block
          when: has_source_scale
          dst: FP8_DYNAMIC
        # 4. Fallback: copy the rest
        - name: fallback
          match: "*"
```

For an already-quantized source, set `src` to the format it is stored in and `dst` to the format you want. To keep the existing quantized values untouched, use `dst: passthrough`.

```bash
qlean --model_name ZhipuAI/GLM-5.3-Flash \
      --model_path /path/to/GLM-5.3-Flash \
      --save_path /path/to/GLM-5.3-Flash-mixed \
      --recipe /path/to/recipe.yaml
```

## Checking What Ran

The log prints a routing report after each run:

```
[region hits] experts: 12288
[region hits] dense: 612
[region hits] fallback: 41
```

Confirm the counts against the tensors you intended. `qlean_config.json` in the output stores the recipe path and its full content, so a run is always traceable to the exact file.

## Full Field Reference

A commented skeleton of every optional field, each shown at its default, lives in
[`examples/EngineFullReference.yaml`](../examples/EngineFullReference.yaml) — that file is
authoritative if it and this page ever disagree.
