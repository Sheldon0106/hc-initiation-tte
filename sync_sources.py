"""Publish explicit HC source snapshots; preserve originals and analysis logic.

Only machine-path declarations and local report links change. A normalized AST comparison
checks that no other executable or statistical code changes in the export.
This command never reads patient caches and never invokes Git or a network API.
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'analysis'))
from hc_paths import configured_path

REPLACEMENTS = {
    'steroid_tte.py': {
        'LOCAL': "LOCAL = configured_path('legacy_workspace', '../restricted/legacy-workspace')",
        'OLD': "OLD = configured_path('legacy_source_dir', '../restricted/legacy-workspace/clinical_tte_20261005')",
        'RAW': "RAW = configured_path('mimic_dir', '../restricted/mimic-iv-3.1')",
        'PRIVATE': "PRIVATE = configured_path('private_dir', '../restricted/hc-private') / 'steroid_tte.pkl.gz'",
        'DEFS': "DEFS = configured_path('sepsis_defs', '../restricted/sepsis3_defs.py')"},
    'steroid_sequential.py': {
        'ROOT': 'ROOT = Path(__file__).resolve().parent',
        'LOCAL': "LOCAL = configured_path('legacy_workspace', '../restricted/legacy-workspace')",
        'STATE': "STATE = configured_path('private_dir', '../restricted/hc-private') / 'steroid_sequential.pkl.gz'",
        'BASE_STATE': "BASE_STATE = configured_path('private_dir', '../restricted/hc-private') / 'steroid_tte.pkl.gz'"},
    'steroid_event_review.py': {
        'ROOT': 'ROOT = Path(__file__).resolve().parent',
        'RAW': "RAW = configured_path('mimic_dir', '../restricted/mimic-iv-3.1')",
        'PRIVATE': "PRIVATE = configured_path('private_dir', '../restricted/hc-private')"},
    'steroid_eicu.py': {
        'ROOT': 'ROOT = Path(__file__).resolve().parent',
        'RAW': "RAW = configured_path('eicu_dir', '../restricted/eicu-crd-2.0')",
        'PRIVATE': "PRIVATE = configured_path('private_dir', '../restricted/hc-private')"}}


def portable_report_paths(text):
    return re.sub(r'C:/Users/[^/\r\n]+/Documents/ChatGPT/TTE/HTE_sepsis/private/',
                  '../restricted/hc-private/', text)


def normalized(text, names):
    tree = ast.parse(portable_report_paths(text))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
                 and not (isinstance(node, ast.ImportFrom) and node.module == 'hc_paths')]
    return ast.dump(tree, include_attributes=False)


def export(text, replacements):
    tree = ast.parse(text); lines = text.splitlines(); edits = []; found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                edits.append((node.lineno - 1, node.end_lineno, replacements[name])); found.add(name)
    assert found == set(replacements), (found, set(replacements))
    for lo, hi, replacement in sorted(edits, reverse=True):
        lines[lo:hi] = [replacement]
    anchor = next(i for i, line in enumerate(lines) if line.startswith('from pathlib import Path'))
    lines.insert(anchor + 1, 'from hc_paths import configured_path')
    result = portable_report_paths('\n'.join(lines) + '\n')
    assert normalized(text, replacements) == normalized(result, replacements)
    return result


def main():
    source = configured_path('canonical_source_dir', '../HTE_sepsis')
    records = []
    for name, substitutions in REPLACEMENTS.items():
        original = (source / name).read_bytes()
        text = original.decode('utf-8-sig').replace('\r\n', '\n')
        result = export(text, substitutions)
        for prohibited in ['C:\\Users\\', 'C:/Users/', 'F:\\2025summer\\', 'F:/2025summer/']:
            assert prohibited not in result, (name, 'machine_path_remaining')
        target = REPO / 'analysis' / name
        target.write_text(result, encoding='utf-8', newline='') if sys.version_info >= (3, 10) else target.write_bytes(result.encode('utf-8'))
        records.append(dict(file=name, original_sha256=hashlib.sha256(original).hexdigest(),
                            exported_sha256=hashlib.sha256(result.encode('utf-8')).hexdigest(), nonpath_AST_equal=True))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
