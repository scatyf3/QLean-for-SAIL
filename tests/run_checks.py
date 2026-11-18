#!/usr/bin/env python3
# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def source_manifest(root):
    root = Path(root).resolve()
    git = ['git', '-c', 'safe.directory=' + str(root), '-C', str(root)]
    names = set(filter(None, subprocess.check_output(git + ['ls-files', '-z']).decode().split('\0')))
    for directory in ('src', 'tests', 'ci', '.github'):
        for p in (root / directory).rglob('*'):
            if (p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
                    and not any(part.endswith('.egg-info') for part in p.parts)
                    and p.suffix not in ('.pyc', '.pyo')):
                names.add(p.relative_to(root).as_posix())
    for pattern in ('*.ini', '*.toml', '*.cfg', '*.py', 'MANIFEST.in', '*requirements*.txt', '*constraints*.txt'):
        names.update(p.name for p in root.glob(pattern) if p.is_file())
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(names) if (root / name).is_file()}


def require_single_card(count):
    if count != 1:
        raise RuntimeError(f'PPU suite requires exactly one visible device; found {count}. Allocate the device explicitly.')


def package_manifest(package):
    package = Path(package)
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}


def execution_errors(data, allowed_failures=None):
    allowed = allowed_failures or {}
    errors = []
    selected = data.get('selected', [])
    if not selected or len(set(selected)) != len(selected):
        errors.append('Selection is empty or contains duplicate nodes')
    if selected != data.get('expected_selected'):
        errors.append('Selected nodes differ from the expected selection')
    if Counter(data.get('deselected', [])) != Counter(data.get('expected_deselected', [])):
        errors.append('Unexpected deselected nodes')
    if Counter(selected + data.get('deselected', [])) != Counter(data.get('collected', [])):
        errors.append('Selected and deselected nodes do not partition the collection')
    reports = data.get('reports', [])
    expected = Counter((node, phase) for node in selected for phase in ('setup', 'call', 'teardown'))
    if Counter((r['nodeid'], r['when']) for r in reports) != expected:
        errors.append('Execution phases are missing, duplicated, or refer to unselected nodes')
    failures = {}
    for report in reports:
        if report.get('wasxfail') is not None or report['outcome'] == 'skipped':
            errors.append('Unexpected skip or xfail')
        elif report['outcome'] == 'failed':
            node = report['nodeid']
            failures[node] = report
            if (report['when'] != 'call' or node not in allowed
                    or not allowed[node] or allowed[node] not in report.get('detail', '')):
                errors.append('Unexpected failure: ' + node)
        elif report['outcome'] != 'passed':
            errors.append('Unknown test outcome')
    if set(failures) != set(allowed):
        errors.append('Failed nodes differ from the explicitly allowed failures')
    if data.get('pytest_exitcode') != (1 if allowed else 0):
        errors.append('pytest exit code does not match the expected completed execution')
    if data.get('collection_errors'):
        errors.append('Test collection failed')
    return errors


def verify_record(data, root, package=None):
    errors = execution_errors(data)
    errors.extend(data.get('audit_errors', []))
    if data.get('schema_version') != 2:
        errors.append('Legacy report lacks the evidence required by the current audit schema')
    current = source_manifest(root)
    if current != data.get('source_manifest_before') or current != data.get('source_manifest_after'):
        errors.append('Source, test, or configuration manifest changed')
    if data.get('installed'):
        if package is None:
            errors.append('Installed report verification requires --package-root')
        else:
            actual = package_manifest(package)
            if actual != data.get('package_before') or actual != data.get('package_after'):
                errors.append('Installed package manifest changed')
    if not data.get('success'):
        errors.append('The original run was not successful')
    return errors


def worker(args):
    import faulthandler
    import pytest
    faulthandler.enable()
    if not args.installed:
        sys.path.insert(0, str(ROOT / 'src'))
    import qlean
    package = Path(qlean.__file__).resolve().parent
    expected = Path(sys.prefix).resolve() if args.installed else (ROOT / 'src').resolve()
    if not package.is_relative_to(expected):
        raise RuntimeError('Unexpected QLean import origin: ' + str(package))
    if args.installed and (sys.prefix == sys.base_prefix or os.environ.get('PYTHONPATH')):
        raise RuntimeError('Installed checks require an isolated venv and an unset PYTHONPATH')
    start = time.monotonic()
    git = ['git', '-c', 'safe.directory=' + str(ROOT), '-C', str(ROOT)]
    data = dict(schema_version=2, suite=args.suite, installed=args.installed,
                base_sha=subprocess.check_output(git + ['rev-parse', 'HEAD']).decode().strip(),
                origin=qlean.__file__, expected_origin=str(expected), python=sys.version, executable=sys.executable,
                versions={n: metadata.version(n) for n in ('torch', 'triton', 'numpy', 'pytest')},
                visibility={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PPU_VISIBLE_DEVICES')},
                source_manifest_before=source_manifest(ROOT), package_before=package_manifest(package),
                collected=[], selected=[], deselected=[], reports=[], collection_errors=[], audit_errors=[])

    class Audit:
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            data['collected'] = [item.nodeid for item in items]
            chosen, excluded = [], []
            for item in items:
                cpu, gpu = (item.get_closest_marker(m) is not None for m in ('cpu', 'gpu'))
                layers = sum(item.get_closest_marker(m) is not None for m in ('unit', 'integration', 'e2e'))
                if cpu == gpu or layers != 1:
                    data['audit_errors'].append('Missing or conflicting test markers: ' + item.nodeid)
                choose = (args.suite == 'cpu' and cpu or args.suite == 'ppu' and gpu)
                (chosen if choose else excluded).append(item)
            data['expected_selected'] = [i.nodeid for i in chosen]
            data['expected_deselected'] = [i.nodeid for i in excluded]
            items[:] = chosen
            config.hook.pytest_deselected(items=excluded)

        def pytest_deselected(self, items):
            data['deselected'].extend(i.nodeid for i in items)

        def pytest_collection_finish(self, session):
            data['selected'] = [i.nodeid for i in session.items]

        def pytest_collectreport(self, report):
            if report.failed:
                data['collection_errors'].append(str(report.longrepr))

        def pytest_runtest_logreport(self, report):
            data['reports'].append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                                        duration=report.duration, wasxfail=getattr(report, 'wasxfail', None),
                                        detail=str(report.longrepr) if report.outcome != 'passed' else ''))

    options = [str(ROOT / 'tests'), '-c', str(ROOT / 'pytest.ini'), '-p', 'no:cacheprovider', '-ra', '--tb=short',
               '--basetemp=' + str(args.output / 'tmp'), '--junitxml=' + str(args.output / 'junit.xml')]
    if args.suite == 'ppu':
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('PPU suite requires the validated Python 3.12 environment')
        constraints = ROOT / 'ci/ppu-constraints.txt'
        data['dependency_versions'] = {}
        for line in constraints.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            name, required = line.strip().split('==')
            actual = metadata.version(name)
            data['dependency_versions'][name] = actual
            if actual != required:
                raise RuntimeError(f'PPU dependency mismatch: {name}={actual}; required {required}')
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        require_single_card(count)
        data['hardware'] = [str(torch.cuda.get_device_properties(0))]
        options.append('--require-gpu')
    data['pytest_exitcode'] = int(pytest.main(options, plugins=[Audit()]))
    data['audit_errors'].extend(execution_errors(data))
    data['source_manifest_after'] = source_manifest(ROOT)
    data['package_after'] = package_manifest(package)
    if data['source_manifest_before'] != data['source_manifest_after']:
        data['audit_errors'].append('Input manifest changed during execution')
    if data['package_before'] != data['package_after']:
        data['audit_errors'].append('QLean package changed during execution')
    imported = {n: m.__file__ for n, m in list(sys.modules.items())
                if (n == 'qlean' or n.startswith('qlean.')) and getattr(m, '__file__', None)}
    data['imported_modules'] = imported
    for name, path in imported.items():
        if not Path(path).resolve().is_relative_to(expected):
            data['audit_errors'].append('Unexpected import origin: ' + name)
    relevant = [r for r in data['reports'] if r['when'] == 'call' or r['outcome'] != 'passed']
    data['counts'] = {key: sum(r['outcome'] == key for r in relevant) for key in ('passed', 'failed', 'skipped')}
    data['counts'].update(collected=len(data['collected']), selected=len(data['selected']), deselected=len(data['deselected']))
    data['seconds'] = round(time.monotonic() - start, 3)
    data['success'] = data['pytest_exitcode'] == 0 and not data['audit_errors']
    save(args.output / 'result.json', data)
    return 0 if data['success'] else 1


def main():
    parser = argparse.ArgumentParser(description='Run audited CPU or single-device PPU tests, or verify an existing report.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--suite', choices=['cpu', 'ppu'])
    mode.add_argument('--verify-report', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_report:
        errors = verify_record(json.loads(args.verify_report.read_text()), args.source_root, args.package_root)
        print(json.dumps({'success': not errors, 'errors': errors}, ensure_ascii=False, indent=2))
        return 1 if errors else 0
    if args.output is None:
        parser.error('--suite requires --output')
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error('Output must be outside the source repository')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON') and k != 'PYTEST_ADDOPTS'}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', TRITON_CACHE_DIR=str(args.output / 'triton_cache'),
               HF_HOME=str(args.output / 'hf_cache'), TORCH_HOME=str(args.output / 'torch_cache'))
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--worker', '--suite', args.suite,
               '--output', str(args.output)] + (['--installed'] if args.installed else [])
    start = time.monotonic()
    with (args.output / 'pytest.log').open('w') as log:
        try:
            proc = subprocess.run(command, cwd=args.output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
    save(args.output / 'execution.json', dict(command=command, returncode=code, cwd=str(args.output),
                                            seconds=round(time.monotonic() - start, 3)))
    print('Results:', args.output, 'Exit code:', code)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
