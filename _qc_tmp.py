from pathlib import Path
import json
import sys
from statistics import median

import pyarrow.parquet as pq
from farkle.utils.manifest import iter_manifest

BASE = Path('data/results_seed_pair_4444_4445')
issues = []
summary = {}

def log_issue(msg):
    issues.append(msg)

def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        log_issue(f"JSON parse failed: {path}: {e}")
        return {}

def check_jsonl_run_manifest(path, label):
    if not path.exists():
        log_issue(f"Missing {label} manifest: {path}")
        return
    has_run_end = False
    bad_steps = []
    error_events = []
    for rec in iter_manifest(path):
        if not isinstance(rec, dict):
            continue
        event = rec.get('event')
        if event == 'run_end':
            has_run_end = True
        if event == 'step_end' and rec.get('ok') is False:
            bad_steps.append(str(rec.get('step') or rec))
        if event in {'step_error', 'run_error'}:
            error_events.append(str(rec.get('step') or rec))
        if rec.get('ok') is False and event not in {'step_end'}:
            error_events.append(str(rec))
    if not has_run_end:
        log_issue(f"{label} manifest missing run_end event: {path}")
    if bad_steps:
        log_issue(f"{label} manifest has failed steps: {path}: {bad_steps[:5]}")
    if error_events:
        log_issue(f"{label} manifest has error events: {path}: {error_events[:5]}")

def schema_sig(schema):
    return tuple((name, str(field.type)) for name, field in zip(schema.names, schema))

def check_parquet(path, label, min_rows=1):
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        log_issue(f"Parquet read failed ({label}): {path}: {e}")
        return None
    md = pf.metadata
    rows = md.num_rows
    if min_rows is not None and rows < min_rows:
        log_issue(f"Parquet has too few rows ({label}): {path} rows={rows}")
    return {
        'rows': rows,
        'columns': md.num_columns,
        'schema': schema_sig(pf.schema_arrow),
    }

def check_rows_dir(row_dir, expected_players):
    result = {
        'row_dir': str(row_dir),
        'manifest_entries': 0,
        'files_on_disk': 0,
        'manifest_rows': 0,
        'parquet_rows': 0,
        'schema_mismatches': 0,
        'row_mismatches': 0,
        'missing_files': 0,
        'extra_files': 0,
        'dup_paths': 0,
        'dup_shuffles': 0,
    }

    manifest_path = row_dir / 'manifest.jsonl'
    if not manifest_path.exists():
        log_issue(f"Missing row manifest: {manifest_path}")
        return result

    manifest_map = {}
    dup_paths = 0
    dup_shuffles = 0
    seen_shuffle = set()
    for rec in iter_manifest(manifest_path):
        if not isinstance(rec, dict):
            continue
        result['manifest_entries'] += 1
        path = rec.get('path')
        if not path:
            log_issue(f"Manifest entry missing path: {manifest_path}: {rec}")
            continue
        if path in manifest_map:
            dup_paths += 1
        manifest_map[path] = rec
        rows_val = rec.get('rows')
        if rows_val is not None:
            try:
                result['manifest_rows'] += int(rows_val)
            except Exception:
                log_issue(f"Invalid rows value in manifest: {manifest_path}: {rec}")
        n_players = rec.get('n_players')
        if n_players is not None and int(n_players) != expected_players:
            log_issue(
                f"Wrong n_players in manifest {manifest_path}: expected {expected_players}, got {n_players}"
            )
        shuffle = rec.get('shuffle_seed')
        if shuffle is not None:
            try:
                shuffle = int(shuffle)
            except Exception:
                shuffle = None
            if shuffle is not None:
                if shuffle in seen_shuffle:
                    dup_shuffles += 1
                seen_shuffle.add(shuffle)

    result['dup_paths'] = dup_paths
    result['dup_shuffles'] = dup_shuffles

    files = [p for p in row_dir.glob('*.parquet') if p.is_file()]
    result['files_on_disk'] = len(files)
    file_set = {p.name for p in files}

    missing_files = [p for p in manifest_map.keys() if p not in file_set]
    extra_files = [p for p in file_set if p not in manifest_map]
    if missing_files:
        log_issue(f"Missing parquet files for manifest {manifest_path}: {missing_files[:5]}")
    if extra_files:
        log_issue(f"Extra parquet files not in manifest {manifest_path}: {extra_files[:5]}")
    result['missing_files'] = len(missing_files)
    result['extra_files'] = len(extra_files)

    schema_ref = None
    schema_mismatches = 0
    row_mismatches = 0
    parquet_rows = 0
    size_bytes = []
    small_files = 0

    for p in files:
        meta = check_parquet(p, label=f"rows {row_dir.name}", min_rows=1)
        if meta is None:
            continue
        parquet_rows += int(meta['rows'])
        if schema_ref is None:
            schema_ref = meta['schema']
        elif meta['schema'] != schema_ref:
            schema_mismatches += 1
        exp = manifest_map.get(p.name, {}).get('rows')
        if exp is not None:
            try:
                exp_i = int(exp)
                if int(meta['rows']) != exp_i:
                    row_mismatches += 1
            except Exception:
                pass
        size = p.stat().st_size
        size_bytes.append(size)
        if size < 1024:
            small_files += 1

    if schema_mismatches:
        log_issue(f"Schema mismatch in {row_dir}: {schema_mismatches} files differ")
    if row_mismatches:
        log_issue(f"Row count mismatch vs manifest in {row_dir}: {row_mismatches} files")
    if small_files:
        log_issue(f"Small row parquet files in {row_dir}: {small_files} files < 1KB")

    result['schema_mismatches'] = schema_mismatches
    result['row_mismatches'] = row_mismatches
    result['parquet_rows'] = parquet_rows

    if size_bytes:
        size_bytes_sorted = sorted(size_bytes)
        med = median(size_bytes_sorted)
        result['size_min'] = min(size_bytes_sorted)
        result['size_median'] = int(med)
        result['size_max'] = max(size_bytes_sorted)
    return result


def check_done_jsons(root):
    for p in root.rglob('*.done.json'):
        data = read_json(p)
        if not data:
            continue
        for key in ('inputs', 'outputs'):
            items = data.get(key)
            if not items:
                continue
            for item in items:
                if not isinstance(item, str):
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = Path.cwd() / path
                if not path.exists():
                    log_issue(f"Done file references missing {key} path: {p}: {item}")

def check_seed(seed_dir, expected_players):
    name = seed_dir.name
    seed_summary = {}

    active_cfg = seed_dir / 'active_config.yaml'
    if not active_cfg.exists():
        log_issue(f"Missing active_config.yaml: {active_cfg}")

    strategy_manifest = seed_dir / 'strategy_manifest.parquet'
    if not strategy_manifest.exists():
        log_issue(f"Missing strategy_manifest.parquet: {strategy_manifest}")
    else:
        meta = check_parquet(strategy_manifest, label=f"{name} strategy_manifest", min_rows=1)
        if meta:
            cols = [c for c, _ in meta['schema']]
            if 'strategy_id' not in cols or 'strategy_str' not in cols:
                log_issue(f"Strategy manifest missing required columns: {strategy_manifest}")

    analysis_manifest = seed_dir / 'analysis' / 'manifest.jsonl'
    if analysis_manifest.exists():
        check_jsonl_run_manifest(analysis_manifest, label=f"{name} analysis")
    else:
        log_issue(f"Missing analysis manifest: {analysis_manifest}")

    combine_manifest = seed_dir / 'analysis' / '02_combine' / 'pooled' / 'all_ingested_rows.manifest.jsonl'
    combine_parquet = seed_dir / 'analysis' / '02_combine' / 'pooled' / 'all_ingested_rows.parquet'
    if combine_parquet.exists() and combine_manifest.exists():
        try:
            manifest_rows = 0
            for rec in iter_manifest(combine_manifest):
                if not isinstance(rec, dict):
                    continue
                rows_val = rec.get('rows') or rec.get('row_count')
                if rows_val is None:
                    continue
                manifest_rows += int(rows_val)
            parquet_rows = pq.ParquetFile(combine_parquet).metadata.num_rows
            if parquet_rows != manifest_rows:
                log_issue(
                    f"Combined parquet rows != manifest rows: {combine_parquet} {parquet_rows} != {manifest_rows}"
                )
            seed_summary['combined_rows'] = parquet_rows
        except Exception as e:
            log_issue(f"Combine manifest check failed: {combine_parquet}: {e}")
    else:
        if not combine_parquet.exists():
            log_issue(f"Missing combined parquet: {combine_parquet}")
        if not combine_manifest.exists():
            log_issue(f"Missing combined manifest: {combine_manifest}")

    metrics_parquet = seed_dir / 'analysis' / '03_metrics' / 'pooled' / 'metrics.parquet'
    if metrics_parquet.exists():
        check_parquet(metrics_parquet, label=f"{name} metrics", min_rows=1)
    else:
        log_issue(f"Missing metrics parquet: {metrics_parquet}")

    for n in expected_players:
        n_dir = seed_dir / f"{n}_players"
        if not n_dir.exists():
            log_issue(f"Missing player dir: {n_dir}")
            continue
        row_dir = n_dir / f"{n}p_rows"
        if row_dir.exists():
            row_summary = check_rows_dir(row_dir, expected_players=n)
            seed_summary[f"{n}p_rows"] = row_summary
        else:
            log_issue(f"Missing rows dir: {row_dir}")

        ckpt_parquet = n_dir / f"{n}p_checkpoint.parquet"
        if ckpt_parquet.exists():
            check_parquet(ckpt_parquet, label=f"{name} {n}p checkpoint", min_rows=1)
        else:
            log_issue(f"Missing checkpoint parquet: {ckpt_parquet}")

        ckpt_pkl = n_dir / f"{n}p_checkpoint.pkl"
        if not ckpt_pkl.exists():
            log_issue(f"Missing checkpoint pkl: {ckpt_pkl}")
        else:
            if ckpt_pkl.stat().st_size == 0:
                log_issue(f"Empty checkpoint pkl: {ckpt_pkl}")

        metrics_pq = n_dir / f"{n}p_metrics.parquet"
        if metrics_pq.exists():
            check_parquet(metrics_pq, label=f"{name} {n}p metrics", min_rows=1)
        else:
            log_issue(f"Missing metrics parquet: {metrics_pq}")

    summary[name] = seed_summary


def check_interseed(root):
    check_done_jsons(root)
    for p in root.rglob('*.json'):
        if p.name.endswith('.jsonl'):
            continue
        _ = read_json(p)
    for p in root.rglob('*.parquet'):
        if any(part.endswith('p_rows') for part in p.parts):
            continue
        check_parquet(p, label='interseed', min_rows=1)


if not BASE.exists():
    print(f"Base directory missing: {BASE}")
    sys.exit(1)

check_jsonl_run_manifest(BASE / 'two_seed_pipeline_manifest.jsonl', label='two_seed_pipeline')

for seed in ('results_seed_4444', 'results_seed_4445'):
    check_seed(BASE / seed, expected_players=[4, 5])

check_interseed(BASE / 'interseed_analysis')

print('QC summary')
print('==========')
for seed_name, seed_info in summary.items():
    print(f"[{seed_name}]")
    for key, val in seed_info.items():
        if isinstance(val, dict) and 'row_dir' in val:
            print(
                f"  {key}: files={val.get('files_on_disk')}, manifest={val.get('manifest_entries')}, "
                f"parquet_rows={val.get('parquet_rows')}, manifest_rows={val.get('manifest_rows')}, "
                f"schema_mismatch={val.get('schema_mismatches')}, row_mismatch={val.get('row_mismatches')}, "
                f"missing_files={val.get('missing_files')}, extra_files={val.get('extra_files')}, "
                f"dup_paths={val.get('dup_paths')}, dup_shuffles={val.get('dup_shuffles')}, "
                f"size_min/med/max={val.get('size_min')}/{val.get('size_median')}/{val.get('size_max')}"
            )
        else:
            print(f"  {key}: {val}")

if issues:
    print('\nIssues')
    print('------')
    for i in issues:
        print(f"- {i}")
    sys.exit(2)
else:
    print('\nNo issues found.')

