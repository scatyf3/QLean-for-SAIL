# Numerical Verification (`verify: bound`)

For supported quantized tensors, QLean compares reconstruction errors against the theoretical bounds of the target formats, allowing a relative tolerance of `1e-4` for float32 noise.

Numerical violations and caught numerical-check errors are recorded without stopping conversion. Report I/O failures are logged without stopping conversion, but may leave reports incomplete. Verification is **not** a substitute for end-to-end accuracy evaluation.

## Turning It On

| Where | How |
|:---|:---|
| Command line | `--verify bound` (overrides the recipe); `--verify off` turns a recipe setting back off |
| Recipe | `verify: bound` under `GeneralQuantEngineModifier` |

Applies only to recipes using `GeneralQuantEngineModifier`. With any other modifier the CLI prints

```
--verify bound was given, but the recipe has no GeneralQuantEngineModifier; ignoring
```

and writes nothing.

## Reports

All files land in `<save_path>/qlean_verify/`:

| File | Content |
|:---|:---|
| `<shard>.csv` | One row per tensor: `numel`, `violations`, `boundary`, `max_over_ratio`, `status`, `detail` |
| `verify_summary.csv` | Per scheme: tensor counts by status, total violations, worst `max_over_ratio` and its tensor |
| `verify_run_info.json` | Shard coverage, row totals, and the cross-check against the run's own statistics |

The directory is created when the first shard is processed and the summary files are written at the end of the run, so an interrupted run can leave partial or no reports.

## Coverage

Verified: 2D `W8A8`, `FP8_DYNAMIC` and `MXFP4` operations.

Other schemes and unsupported dimensions are recorded as `skipped_scheme` or `skipped_<N>d`.

Other schemes are recorded as `skipped`.

## Reading the Result

The run ends with a verdict:

```
[verify] Coverage: 141/141 shards
[verify] Result: PASSED | Checked 141/141 shards, 59044 rows = ok 59044 / violation 0 / skipped 0 / error 0
```

| Status | Meaning | Action |
|:---|:---|:---|
| `ok` | Error within bound | None |
| `violation` | At least one element exceeds bound × tolerance | Investigate; a high `boundary` count in other rows means results sit right at the threshold |
| `error` | Verification itself failed (OOM, shape mismatch, …) | Fix the cause — the row carries no verdict on the quantization |
| `skipped` | Not verified (see coverage above) | Confirm it matches expectations |

`<shard>.csv` carries the specific reason in `status` (`skipped_scheme`, `skipped_int4_nonabsmax`, `skipped_3d`, `error_<Exception>`); the four names above are the aggregation used in `verify_summary.csv`.

Any `violation` row, `error` row, or missing shard report makes the result `FAILED`. `skipped` rows do not.

## Verification and Resume

With verification enabled, a shard is reused only if its verification records are complete. Shards with no records (produced while verification was off), with error rows, or with a truncated report are **fully rerun — re-quantized and re-verified**, not just re-verified.

Switching a completed run from `off` to `bound` therefore reprocesses every shard.

## Troubleshooting

| Symptom | Cause |
|:---|:---|
| No `qlean_verify/` and the "ignoring" WARNING above | The matched recipe uses a legacy modifier. Point `--recipe` at one with `GeneralQuantEngineModifier`. |
| `[verify] Bound checks enabled; details written to <path>` but no directory yet | Normal before the first shard finishes; `<path>` in that line is authoritative. |
| No `[verify]` line at all | The installed build predates the feature. Check `qlean --help \| grep -- --verify`, then reinstall. |
| `FAILED` with `error` rows only | Verification ran out of memory or hit a shape mismatch; the quantized output itself is still written. Lower `num_gpus` or rerun the affected shards. |
