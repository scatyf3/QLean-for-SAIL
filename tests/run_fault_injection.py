#!/usr/bin/env python3
# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import difflib
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
AUDIT = runpy.run_path(str(ROOT / 'tests/run_checks.py'), run_name='fault_audit')
CHECKPOINT = 'tests/integration/test_extended_formats_checkpoint.py'
CASES = {
    'scale': dict(
        path='src/qlean/modifiers/utils.py',
        before='    return quantized.to(torch.int8), scale.to(torch.float32)',
        after='    return quantized.to(torch.int8), scale.to(torch.float32) * 2',
        node='tests/modifiers/test_unit_w8a8_numerics.py::test_w8a8_codes_scales_and_rounding[matrix-float32-cpu]',
        function='test_w8a8_codes_scales_and_rounding',
        check='assert_tensor_equal(scale, expected_scale,'),
    'nibbles': dict(
        path='src/qlean/utils/pack_acext_int4.py',
        before='    return (low_nibbles | (high_nibbles << 4)).view(torch.int8)',
        after='    return (high_nibbles | (low_nibbles << 4)).view(torch.int8)',
        node='tests/utils/test_unit_int4_layout.py::test_nibble_order_and_signed_endpoints',
        function='test_nibble_order_and_signed_endpoints',
        check='torch.testing.assert_close(packed.view(torch.uint8), expected,'),
    'index': dict(
        path='src/qlean/modifiers_engine/_runtime/engine.py',
        before='            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)',
        after='            shards = sorted(set(weight_map.values()))\n'
              '            wrong_map = {name: shards[(shards.index(shard) + 1) % len(shards)]\n'
              '                         for name, shard in weight_map.items()}\n'
              '            json.dump({"metadata": {}, "weight_map": wrong_map}, f, indent=2)',
        node=CHECKPOINT + '::test_packed_source_requantization_checkpoint[int4-W8A8]',
        function='_run_and_check',
        check='assert index["weight_map"] == observed_map'),
    'stream_tail': dict(
        path='src/qlean/modifiers_engine/_runtime/plan.py',
        before='    for a in range(0, M, chunk_rows):\n        b = min(a + chunk_rows, M)',
        after='    for a in range(0, M, chunk_rows):\n'
              '        if out_buf is not None and a + chunk_rows >= M:\n'
              '            out_buf[a:].zero_()\n'
              '            scale_buf[a:].zero_()\n'
              '            break\n'
              '        b = min(a + chunk_rows, M)',
        node=CHECKPOINT + '::test_mxfp8_streaming_checkpoint[FP8_DYNAMIC]',
        function='_run_and_check',
        check='torch.testing.assert_close(actual[name].contiguous().reshape(-1).view(torch.uint8),'),
    'region_order': dict(
        path='src/qlean/modifiers_engine/_runtime/region_router.py',
        before='    for region in regions:\n        if not _match_pattern(name, region.get("match", "*")):',
        after='    for region in reversed(regions):\n        if not _match_pattern(name, region.get("match", "*")):',
        node='tests/modifiers_engine/test_unit_region_plan.py::test_first_match_drives_op_and_quantizer'
             '[False-False-float32-underscore-model.proj.weight_scale]',
        function='test_first_match_drives_op_and_quantizer',
        check='assert result.meta["region"] == "specific"'),
}


def snapshot(root):
    result = {}
    for relative in ('src/qlean', 'tests', 'pytest.ini'):
        base = root / relative
        paths = [base, *sorted(base.rglob('*'))] if base.is_dir() else [base]
        for path in paths:
            if any(part in ('__pycache__', '.pytest_cache') for part in path.relative_to(root).parts):
                continue
            if path.is_symlink():
                raise ValueError(f'Isolated inputs must not contain symlinks: {path}')
            if path.is_dir() or path.suffix in ('.pyc', '.pyo'):
                continue
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def copy_inputs(root, destination, manifest):
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    if snapshot(destination) != manifest:
        raise RuntimeError('Inputs changed during copying')


def mutate_text(text, case):
    if case['before'] == case['after'] or text.count(case['before']) != 1:
        raise ValueError('Mutation must match exactly once and change the text')
    mutated = text.replace(case['before'], case['after'], 1)
    compile(mutated, case['path'], 'exec')
    return mutated


def assertion_site(root, case):
    relative = case['node'].split('::')[0]
    text = (root / relative).read_text(encoding='utf-8')
    functions = [node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name == case['function']]
    if len(functions) != 1:
        raise ValueError('Expected exactly one assertion function')
    statements = [node for node in ast.walk(functions[0]) if isinstance(node, (ast.Assert, ast.Expr))
                  and ast.get_source_segment(text, node).strip().startswith(case['check'])]
    if len(statements) != 1:
        raise ValueError('Expected exactly one assertion statement')
    statement, = statements
    return dict(file=relative, function=case['function'], start=statement.lineno, end=statement.end_lineno)


def classify(data, case, site, mutated):
    if data.get('audit_errors') or data.get('process_returncode') != data.get('pytest_exitcode'):
        return 'ERROR'
    if data.get('expected_selected') != [case['node']]:
        return 'ERROR'
    if not AUDIT['execution_errors'](data):
        return 'SURVIVED' if mutated else 'BASELINE_PASSED'
    if not mutated:
        return 'BASELINE_FAILED'
    if AUDIT['execution_errors'](data, {case['node']: 'AssertionError'}):
        return 'ERROR'
    call, = [report for report in data['reports'] if report['when'] == 'call']
    at_check = any(frame['file'] == site['file'] and frame['function'] == site['function']
                   and site['start'] <= frame['line'] <= site['end'] for frame in call.get('frames', []))
    return 'KILLED' if call.get('exception') == 'builtins.AssertionError' and at_check else 'ERROR'


def worker(args):
    case_id, = args.case
    if ROOT != args.output.parent / 'source' or args.output.name not in ('baseline', 'mutant'):
        raise ValueError('Worker must run in an isolated copy created by this runner')
    marker = json.loads((ROOT / '.fault-copy.json').read_text(encoding='utf-8'))
    if marker['case'] != case_id or ROOT == Path(marker['original_root']).resolve():
        raise ValueError('Isolated copy marker mismatch')
    import faulthandler
    import pytest

    faulthandler.enable()
    sys.path.insert(0, str(ROOT / 'src'))
    import qlean

    case = CASES[case_id]
    before = snapshot(ROOT)
    data = dict(expected_selected=[case['node']], expected_deselected=[], collected=[], selected=[],
                deselected=[], reports=[], collection_errors=[], audit_errors=[],
                origin=qlean.__file__, python=sys.version, executable=sys.executable,
                manifest_before=before, visibility={key: os.environ.get(key) for key in
                                                   ('CUDA_VISIBLE_DEVICES', 'PPU_VISIBLE_DEVICES')})
    expected_package = ROOT / 'src/qlean'
    if Path(qlean.__file__).resolve() != expected_package / '__init__.py':
        raise RuntimeError('qlean was not imported from the isolated copy')

    class Recorder:
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            data['collected'] = [item.nodeid for item in items]

        def pytest_collection_finish(self, session):
            data['selected'] = [item.nodeid for item in session.items]

        def pytest_deselected(self, items):
            data['deselected'].extend(item.nodeid for item in items)

        def pytest_collectreport(self, report):
            if report.failed:
                data['collection_errors'].append(str(report.longrepr))

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            report = (yield).get_result()
            frames = []
            if call.excinfo is not None:
                for entry in call.excinfo.traceback:
                    path = Path(str(entry.path)).resolve()
                    if path.is_relative_to(ROOT):
                        frames.append(dict(file=path.relative_to(ROOT).as_posix(),
                                           function=entry.name, line=entry.lineno + 1))
            report.fault_frames = frames
            report.fault_exception = (f'{call.excinfo.type.__module__}.{call.excinfo.type.__qualname__}'
                                      if call.excinfo is not None else None)

        def pytest_runtest_logreport(self, report):
            data['reports'].append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                                        wasxfail=getattr(report, 'wasxfail', None),
                                        frames=getattr(report, 'fault_frames', []),
                                        exception=getattr(report, 'fault_exception', None),
                                        detail=str(report.longrepr) if report.outcome != 'passed' else ''))

    options = [case['node'], '-c', str(ROOT / 'pytest.ini'), '--rootdir=' + str(ROOT),
               '-p', 'no:cacheprovider', '--require-gpu', '-v', '-ra', '--tb=long',
               '--basetemp=' + str(args.output / 'tmp'), '--junitxml=' + str(args.output / 'junit.xml')]
    data['pytest_exitcode'] = int(pytest.main(options, plugins=[Recorder()]))
    data['manifest_after'] = snapshot(ROOT)
    imported = {name: str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
                if (name == 'qlean' or name.startswith('qlean.')) and getattr(module, '__file__', None)}
    data['imported_modules'] = imported
    if any(not Path(path).is_relative_to(expected_package) for path in imported.values()):
        data['audit_errors'].append('qlean modules were imported from outside the isolated copy')
    if str(ROOT / case['path']) not in imported.values():
        data['audit_errors'].append('Mutation target module was not imported')
    data['versions'] = {}
    for name in ('pytest', 'torch', 'triton', 'safetensors', 'compressed-tensors'):
        try:
            data['versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            data['versions'][name] = None
    AUDIT['save'](args.output / 'result.json', data)
    return data['pytest_exitcode']


def run_worker(source, output, case_id, manifest, timeout):
    output.mkdir(exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PYTHON', 'PYTEST'))}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', TRITON_CACHE_DIR=str(output.parent / 'triton_cache'),
               HF_HOME=str(output / 'hf_cache'), TORCH_HOME=str(output / 'torch_cache'),
               XDG_CACHE_HOME=str(output / 'cache'))
    command = [sys.executable, '-I', '-B', str(source / 'tests/run_fault_injection.py'),
               '--worker', '--case', case_id, '--output', str(output)]
    start = time.monotonic()
    with (output / 'pytest.log').open('w', encoding='utf-8') as log:
        try:
            code = subprocess.run(command, cwd=source, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    result = output / 'result.json'
    data = json.loads(result.read_text(encoding='utf-8')) if result.is_file() else {
        'audit_errors': ['Worker did not produce a complete report; see pytest.log']}
    data.update(process_returncode=code, command=command, seconds=round(time.monotonic() - start, 3))
    if data.get('manifest_before') != manifest or data.get('manifest_after') != manifest:
        data['audit_errors'].append('Isolated copy differs from the expected manifest before or after execution')
    if snapshot(source) != manifest:
        data['audit_errors'].append('Isolated copy changed after worker exit')
    AUDIT['save'](result, data)
    return data


def main():
    parser = argparse.ArgumentParser(description='Check whether existing tests detect five faults in isolated copies.')
    parser.add_argument('--output', type=Path, required=True, help='New directory outside the repository for copies, logs, and reports')
    parser.add_argument('--case', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each pytest subprocess')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.timeout <= 0 or len(set(args.case)) != len(args.case):
        parser.error('Timeout must be positive and fault cases must be unique')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.worker:
        if len(args.case) != 1:
            parser.error('Worker requires exactly one fault case')
        return worker(args)
    if args.output.is_relative_to(ROOT) or ROOT.is_relative_to(args.output):
        parser.error('Output directory must neither be inside nor contain the source repository')
    original = snapshot(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(source_root=str(ROOT), requested_cases=args.case, source_manifest_before=original,
                   cases=[], errors=[], success=False)
    try:
        for case_id in args.case:
            case = CASES[case_id]
            directory = args.output / case_id
            source = directory / 'source'
            row = dict(case=case_id, node=case['node'], status='ERROR')
            summary['cases'].append(row)
            try:
                copy_inputs(ROOT, source, original)
                AUDIT['save'](source / '.fault-copy.json', dict(case=case_id, original_root=str(ROOT)))
                target = source / case['path']
                text = target.read_text(encoding='utf-8')
                mutated = mutate_text(text, case)
                site = assertion_site(source, case)
                row['assertion'] = site
                baseline = run_worker(source, directory / 'baseline', case_id, original, args.timeout)
                row['baseline'] = classify(baseline, case, site, mutated=False)
                if row['baseline'] != 'BASELINE_PASSED':
                    row['status'] = row['baseline']
                else:
                    target.write_text(mutated, encoding='utf-8')
                    patch = ''.join(difflib.unified_diff(text.splitlines(keepends=True),
                                                       mutated.splitlines(keepends=True),
                                                       fromfile=case['path'], tofile=case['path']))
                    (directory / 'mutation.patch').write_text(patch, encoding='utf-8')
                    expected = dict(original)
                    expected[case['path']] = hashlib.sha256(mutated.encode('utf-8')).hexdigest()
                    if snapshot(source) != expected:
                        raise RuntimeError('Mutation changed files outside the target')
                    result = run_worker(source, directory / 'mutant', case_id, expected, args.timeout)
                    row['status'] = classify(result, case, site, mutated=True)
            except Exception:
                row['error'] = traceback.format_exc()
            print(f"{case_id}: {row['status']}", flush=True)
    finally:
        summary['source_manifest_after'] = snapshot(ROOT)
        if summary['source_manifest_after'] != original:
            summary['errors'].append('Workspace inputs changed; no files were reverted')
        summary['success'] = (len(summary['cases']) == len(args.case) and not summary['errors']
                              and all(row['status'] == 'KILLED' for row in summary['cases']))
        AUDIT['save'](args.output / 'summary.json', summary)
        print('Report directory:', args.output, flush=True)
    return 0 if summary['success'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
