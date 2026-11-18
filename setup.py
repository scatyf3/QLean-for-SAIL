# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setuptools import find_packages, setup

README_TEXT = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
setup(
    name="qlean",
    version="0.3.0",
    description=("A toolkit for quantizing and dequantizing large language models."),
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    keywords=(
        "qlean, llms, large language models, transformers, pytorch, "
        "huggingface, compressors, compression, quantization, model compression"
    ),
    license_expression="Apache-2.0",
    license_files=["LICENSE", "NOTICE", "LICENSES/MIT.txt"],
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(
        "src", include=["qlean", "qlean.*"], exclude=["*.__pycache__.*"]
    ),
    install_requires=[
        "pyyaml>=6.0,<7",
        "tqdm>=4.60,<5",
        "torch>=2.7,<2.14",
        "triton>=3.4,<3.8",
        "transformers>=4.54,<6",
        "huggingface-hub>=0.34,<3",
        "safetensors>=0.5.0,<1",
        "pydantic>=2.4,<3",
        "compressed-tensors>=0.13,<0.14",
    ],
    extras_require={
        "dev": [
            # testing framework
            "pytest>=6.0.0",
            # linting, formatting, and type checking
            "black~=24.4.2",
            "isort~=5.13.2",
            "mypy~=1.10.0",
            "ruff~=0.17.3",
            "flake8~=7.0.0",
            # pre commit hooks
            "pre-commit",
        ],
        # hadamard rotation (MXFP4 + rotation=hadamard) needs scipy
        "rotation": ["scipy>=1.11.4,<2"],
    },
    entry_points={
        "console_scripts": [
            "qlean=qlean.cli.__main__:main",
        ]
    },
    python_requires=">=3.10,<3.15",
)
