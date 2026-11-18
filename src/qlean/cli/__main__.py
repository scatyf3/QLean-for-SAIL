# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import re

def setup_logging(level):
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)

_INPUT_FORMAT_DEFS = {
    "FP8":   ["FP8"],
    "NVFP4": ["NVFP4"],
    "BF16":  ["BF16"],
}

INPUT_FORMAT_ALIASES = {}
for canonical, aliases in _INPUT_FORMAT_DEFS.items():
    for alias in aliases:
        for variant in (alias, alias.upper(), alias.lower()):
            INPUT_FORMAT_ALIASES[variant] = canonical

def parse_input_format(value: str) -> str:
    canonical = INPUT_FORMAT_ALIASES.get(value)
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"Unknown input format '{value}'. Supported: FP8, NVF4, BF16"
        )
    return canonical

_OUTPUT_FORMAT_DEFS = {
    "INT8_W8A8":   ["W8A8", "INT8", "INT8_W8A8"],
    "FP8_BLOCK":   ["FP8-BLOCK", "FP8_BLOCK"],
    "FP8_DYNAMIC": ["FP8-DYNAMIC", "FP8_DYNAMIC", "FP8-CHANNEL", "FP8_CHANNEL"],
    "MXFP4":       ["MXFP4"],
    "BF16":        ["BF16"],
}

OUTPUT_FORMAT_ALIASES = {}
for canonical, aliases in _OUTPUT_FORMAT_DEFS.items():
    for alias in aliases:
        for variant in (alias, alias.upper(), alias.lower()):
            OUTPUT_FORMAT_ALIASES[variant] = canonical

def parse_output_format(value: str) -> str:
    canonical = OUTPUT_FORMAT_ALIASES.get(value)
    if canonical is None:
        supported = sorted(set(OUTPUT_FORMAT_ALIASES.values()))
        aliases = {v: [k for k, v2 in OUTPUT_FORMAT_ALIASES.items() if v2 == v] for v in supported}
        lines = [f"  {fmt}: {', '.join(a)}" for fmt, a in aliases.items()]
        raise argparse.ArgumentTypeError(
            f"Unknown output format '{value}'\nSupported formats:\n" + "\n".join(lines)
        )
    return canonical

_DESCRIPTION = """\
\nQLean - A toolkit for quantizing and converting large language models.

How it works:
  QLean auto-detects your model's current format (BF16, FP8, NVFP4),
  then applies the necessary conversion to reach your target format.
  If the source model is already quantized (e.g. FP8), QLean will
  dequantize first, then re-quantize to the target format."""

_USAGE = """\
qlean --model_name MODEL_NAME --model_path MODEL_PATH
             --save_path SAVE_PATH [options]"""

def _print_examples():
    """Print usage examples."""
    print("""\
\nExamples:

  # 1. Quantize a BF16 model to INT8 (default output format)
  qlean --model_name Qwen/Qwen3-235B-A22B \\
        --model_path /path/to/Qwen3-235B-A22B \\
        --save_path /path/to/Qwen3-235B-A22B-INT8

  # 2. FP8 model → INT8 (input format auto-detected from model files)
  qlean --model_name deepseek-ai/DeepSeek-V3.2 \\
        --model_path /path/to/DeepSeek-V3.2 \\
        --save_path /path/to/DeepSeek-V3.2-INT8

  # 3. Choose a different output format with --output-format
  qlean --model_name Qwen/Qwen3-30B-A3B \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --save_path /path/to/Qwen3-30B-A3B-FP8Block \\
        --output-format FP8_BLOCK

  # 4. Use a custom recipe file (overrides built-in recipe)
  qlean --model_name Qwen/Qwen3-8B \\
        --model_path /path/to/Qwen3-8B \\
        --save_path /path/to/Qwen3-8B-custom \\
        --recipe /path/to/recipe.yaml
""")

def _print_formats():
    """Print supported format conversions."""
    from qlean.core.auto_recipe import AUTO_CONVERSIONS
    from collections import defaultdict

    # Group by input format
    conversions = defaultdict(list)
    for (inp, out) in AUTO_CONVERSIONS:
        conversions[inp].append(out)

    print("\nSupported conversions:\n")
    print(f"  {'Input':<10} {'Output Formats'}")
    print(f"  {'─' * 10} {'─' * 40}")
    for inp in sorted(conversions):
        outputs = ", ".join(sorted(conversions[inp]))
        print(f"  {inp:<10} {outputs}")

    output_aliases = {c: [a for a in al if a != c] for c, al in _OUTPUT_FORMAT_DEFS.items()}
    if any(output_aliases.values()):
        print("\nOutput format aliases:")
        for canonical, other in output_aliases.items():
            if other:
                print(f"  {canonical}: {', '.join(other)}")

    input_aliases = {c: [a for a in al if a != c] for c, al in _INPUT_FORMAT_DEFS.items()}
    if any(input_aliases.values()):
        print("\nInput format aliases:")
        for canonical, other in input_aliases.items():
            if other:
                print(f"  {canonical}: {', '.join(other)}")
    print()

def _natural_sort_key(s):
    """Return a key for natural sorting (e.g. 8B < 30B < 235B)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def _print_models():
    """Print models with optimized recipes, grouped by output format."""
    from importlib.resources import files
    from qlean.core.routes import RECIPE_DIR_MAP

    # Build reverse map: dir_name -> list of output formats, excluding BF16
    dir_to_fmts = {}
    for fmt, dir_name in sorted(RECIPE_DIR_MAP.items()):
        if fmt == "BF16":
            continue
        dir_to_fmts.setdefault(dir_name, []).append(fmt)

    print("\nModels with dedicated recipes")
    print("(grouped by supported output format):\n")
    for dir_name, fmts in sorted(dir_to_fmts.items()):
        label = " / ".join(sorted(fmts))
        print(f"  {label}:")

        base = files("qlean.configs").joinpath(dir_name)
        try:
            model_names = []
            for item in base.iterdir():
                name = item.name
                if name.endswith(".yaml") and not name.startswith("_"):
                    model_names.append(name.removesuffix('.yaml'))
        except (FileNotFoundError, TypeError):
            model_names = []

        # Group by brand (first segment before '-'), sort naturally within
        brand_map = {}
        for name in model_names:
            brand = name.split("-")[0]
            brand_map.setdefault(brand, []).append(name)

        max_brand_len = max((len(b) for b in brand_map), default=0)
        for brand in sorted(brand_map):
            models = sorted(brand_map[brand], key=_natural_sort_key)
            padding = ' ' * (max_brand_len - len(brand))
            print(f"    {brand}:{padding}  {', '.join(models)}")
        print()

    print("Models not listed may still work with architecture-level or default")
    print("recipes, which may not be fully optimized for those models.")
    print()
    print("Use --list-models MODEL_NAME to check support for any model.\n")

def _print_model_recipe(model_name: str):
    """Check whether a model has dedicated recipe support."""
    from importlib.resources import files
    from qlean.core.routes import RECIPE_DIR_MAP

    short_name = model_name.split("/")[-1]

    # Build reverse map: dir_name -> list of output formats, excluding BF16
    dir_to_fmts = {}
    for fmt, dir_name in sorted(RECIPE_DIR_MAP.items()):
        if fmt == "BF16":
            continue
        dir_to_fmts.setdefault(dir_name, []).append(fmt)

    matched_fmts = []
    all_fmts = []
    for dir_name, fmts in sorted(dir_to_fmts.items()):
        all_fmts.extend(fmts)
        base = files("qlean.configs").joinpath(dir_name)
        exact = base / f"{short_name}.yaml"
        if exact.is_file():
            matched_fmts.extend(fmts)

    if matched_fmts:
        label = " / ".join(sorted(matched_fmts))
        print(f"\n'{short_name}' has a dedicated recipe for: {label}")
        missing_fmts = sorted(set(all_fmts) - set(matched_fmts))
        if missing_fmts:
            missing_label = " / ".join(missing_fmts)
            print(f"\nNo dedicated recipe for: {missing_label}")
            print("These formats will use a fallback recipe,")
            print("which may not be fully optimized for this model.")
        print()
    else:
        print(f"\nNo dedicated recipe for '{short_name}'.")
        print("QLean will use an architecture-level or default recipe at runtime,")
        print("which may not be fully optimized for this model.\n")

def main():
    parser = argparse.ArgumentParser(
        prog="qlean",
        usage=_USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_DESCRIPTION,
        add_help=False,
    )

    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=None,
        metavar="MODEL_NAME",
        help="Model identifier (e.g. 'Qwen/Qwen3-235B-A22B')",
    )
    required_group.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        metavar="MODEL_PATH",
        help="Path to the source model directory",
    )
    required_group.add_argument(
        "--save_path",
        type=str,
        required=False,
        default=None,
        metavar="SAVE_PATH",
        help="Path to save the output model",
    )

    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--output-format",
        type=parse_output_format,
        default="INT8_W8A8",
        metavar="FORMAT",
        help="Target output format (default: INT8_W8A8)",
    )
    options_group.add_argument(
        "--input-format",
        type=parse_input_format,
        default=None,
        metavar="FORMAT",
        help="Source model format (auto-detected if omitted)",
    )
    options_group.add_argument(
        "--recipe",
        type=str,
        required=False,
        help="Path to a custom recipe YAML file",
    )
    options_group.add_argument(
        "--mix_path",
        type=str,
        required=False,
        metavar="MIX_PATH",
        help="Path to save the mixed-precision model",
    )
    options_group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO)",
    )
    options_group.add_argument(
        "--verify",
        choices=["off", "bound"],
        default=None,
        help="Override GeneralQuantEngineModifier.verify (off|bound); "
             "omitted = follow recipe",
    )

    # Hidden dataset arguments
    options_group.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        help=argparse.SUPPRESS,
    )
    options_group.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help=argparse.SUPPRESS,
    )
    options_group.add_argument(
        "--other_args",
        type=str,
        required=False,
        help=argparse.SUPPRESS,
    )

    info_group = parser.add_argument_group("Info Commands")
    info_group.add_argument(
        "-h", "--help",
        action="help",
        help="Show this help message and exit",
    )
    info_group.add_argument(
        "--list-formats",
        action="store_true",
        help="Show all supported format conversions and aliases",
    )
    info_group.add_argument(
        "--list-models",
        nargs="?",
        const=True,
        default=None,
        metavar="MODEL",
        help="List supported models, or show a model's recipe",
    )
    info_group.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples",
    )

    args = parser.parse_args()

    # Handle info commands (no model args required)
    if getattr(args, 'list_formats', False):
        _print_formats()
        return
    if getattr(args, 'list_models', None) is not None:
        if args.list_models is True:
            _print_models()
        else:
            _print_model_recipe(args.list_models)
        return
    if getattr(args, 'examples', False):
        _print_examples()
        return

    # Validate required args for quantization
    missing = []
    for name in ('model_name', 'model_path', 'save_path'):
        if getattr(args, name) is None:
            missing.append(f'--{name}')
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Application start")

    # Load heavy deps (torch etc.) only when entering the quantization flow;
    # info-only commands stay lightweight.
    from qlean.utils import parse_to_dataclass
    from qlean.entrypoints import post_train

    model_args, dataset_args, other_args = parse_to_dataclass(args)
    post_train(model_args, dataset_args, other_args)

if __name__ == "__main__":
    main()