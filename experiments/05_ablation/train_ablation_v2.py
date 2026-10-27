#!/usr/bin/env python3
"""Standalone SSH/grid trainer for DeepKoopFormer ablation.

Dependencies: torch, numpy; standard-library otherwise.
Model definitions are extracted from 02_ablation_model.ipynb via Python AST.
Data: four .pt bundles exported from 01_ablation_data.ipynb.

Default mode is --dry-run: no training until --run is explicitly supplied.
Completed runs produce a standalone flat .pt in results/<dataset>_grid_search_eN/
while preserving nested resume checkpoints. --export-flat and --watch-flat can
convert completed old runs without interrupting an existing tmux training job.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PATCH_LENGTHS = (70, 80, 90, 100, 110, 120, 130)
HORIZONS = (2, 4, 6, 8, 10, 12, 14, 16)
MODEL_VARIANTS = (
    'DLinear', 'LSTM', 'PatchTST', 'PatchTST+LinearLatent',
    'KoopPatchTST_NoLyap', 'KoopPatchTST_Full',
)
DATASETS = {
    'Electricity': 'electricity_ablation.pt',
    'Financial': 'financial_ablation.pt',
    'CMIP6 Wind': 'cmip6_wind_ablation.pt',
    'CMIP6 Pressure': 'cmip6_pressure_candidate_2500.pt',
}
D_MODEL = 96
D_LATENT = 96
LSTM_HIDDEN_DIM = 96
LSTM_NUM_LAYERS = 2
NUM_ENCODER_LAYERS = 3
NUM_ATTENTION_HEADS = 4
DIM_FEEDFORWARD = 96
RHO_MAX = 0.99
LYAP_WEIGHT = 0.1
SEED = 7
EPOCHS = 4000
LEARNING_RATE = 3e-4

DEFINITIONS = (
    '_orth', 'StrictStableKoopmanOperator', 'LinearLatentPropagator',
    'SinCosPosEnc', 'PatchEmbed1D', 'PatchTST_Backbone',
    'PatchTSTBaseline', 'PatchTST_UnconstrainedLatent',
    'Koopformer_PatchTST', 'SimpleLSTMForecaster',
    'DLinearForecaster', 'create_ablation_model',
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_notebook(path):
    """Load ONLY selected top-level model definitions; never execute notebook cells."""
    with open(path, encoding='utf-8') as handle:
        notebook = json.load(handle)
    found = {}
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = ''.join(source) if isinstance(source, list) else source
        if not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise RuntimeError(f'Model notebook has invalid Python: {exc}') from exc
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in DEFINITIONS:
                if node.name in found:
                    raise ValueError(f'Duplicate model definition: {node.name}')
                found[node.name] = node
    missing = set(DEFINITIONS) - set(found)
    if missing:
        raise ValueError(f'Missing definitions in {path}: {sorted(missing)}')
    # Preserve original declaration order to satisfy references/base classes.
    ordered = []
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue
        s = cell.get('source', '')
        s = ''.join(s) if isinstance(s, list) else s
        tree = ast.parse(s)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in found:
                ordered.append(node)
    namespace = dict(globals())
    # This is a trusted, user-authored notebook. Only classes/functions are evaluated.
    code = ast.fix_missing_locations(ast.Module(body=ordered, type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace


def load_bundle(path):
    try:
        bundle = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        bundle = torch.load(path, map_location='cpu')
    for key in ('train', 'test', 'input_dim'):
        if key not in bundle:
            raise KeyError(f'{path}: missing {key}')
    for key in ('train', 'test'):
        t = bundle[key]
        if not isinstance(t, torch.Tensor) or t.ndim != 2:
            raise ValueError(f'{path}: {key} must be 2-D tensor')
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f'{path}: {key} contains invalid values')
        if t.shape[1] != int(bundle['input_dim']):
            raise ValueError(f'{path}: inconsistent channel count')
    digest = hashlib.sha256()
    for key in ('train', 'test'):
        t = bundle[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.numpy().tobytes())
    return bundle, digest.hexdigest()


def windows(series, p, h):
    """Author's convention: range(len(series)-p-h), feature-major Y."""
    if not isinstance(series, torch.Tensor):
        series = torch.as_tensor(series, dtype=torch.float32)
    series = series.detach().cpu().to(dtype=torch.float32).contiguous()
    n = len(series) - p - h
    if n <= 0:
        raise ValueError(f'Not enough timestamps: {len(series)} for P={p}, H={h}')
    X = torch.stack([series[i:i+p] for i in range(n)])
    Y = torch.stack([series[i+p:i+p+h].T.reshape(-1) for i in range(n)])
    return X, Y


def loss_parts(model, X, Y, lyap_weight):
    pred, z, z_next = model(X, return_latents=True)
    mse = F.mse_loss(pred, Y)
    if lyap_weight:
        # P=I; identical to paper's quadratic Lyapunov regularization.
        lyap = torch.relu(z_next.square().sum(1) - z.square().sum(1)).mean()
    else:
        lyap = mse.new_zeros(())
    return mse + lyap_weight*lyap, mse, lyap


def metrics(model, X, Y, device, batch_size):
    model.eval()
    sum_sq, sum_abs, count = 0., 0., 0
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = X[start:start+batch_size].to(device)
            yb = Y[start:start+batch_size].to(device)
            diff = (model(xb) - yb).double()
            sum_sq += diff.square().sum().item()
            sum_abs += diff.abs().sum().item()
            count += diff.numel()
    if not count:
        raise ValueError('Empty evaluation split')
    return {'MSE': sum_sq/count, 'MAE': sum_abs/count, 'Samples': len(X)}


def snapshot_rng():
    n = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': {'name': n[0], 'keys': n[1].tolist(), 'pos': int(n[2]),
                      'has_gauss': int(n[3]), 'cached': float(n[4])}}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda'):
        if len(state['cuda']) != torch.cuda.device_count():
            print('WARNING: GPU count changed; CUDA RNG exact restoration unavailable', flush=True)
        else:
            torch.cuda.set_rng_state_all(state['cuda'])
    n = state['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32),
                         n['pos'], n['has_gauss'], n['cached']))


def safe_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def atomic_torch_save(obj, path):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def output_id(dataset, model, p, h, epochs, seed):
    slug = lambda s: re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return Path(slug(dataset)) / f'p{p}_h{h}' / slug(model) / f'e{epochs}_seed{seed}'




def flat_output_path(output_dir, cfg):
    """Electricity-style one-file-per-configuration final artifact."""
    slug = lambda s: re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    ds_slug = slug(cfg['dataset'])
    if cfg['dataset'] == 'CMIP6 Pressure':
        ds_slug = 'cmip6_pressure_candidate_2500'
    directory = output_dir.parent / f'{ds_slug}_grid_search_e{cfg["epochs"]}'
    name = f'{slug(cfg["model"])}_p{cfg["p"]}_h{cfg["h"]}_d{cfg["d_model"]}.pt'
    return directory / name


def predict_cpu_tensor(model, X, device, batch_size):
    """Deterministic full-split predictions saved for plotting like Electricity."""
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            chunks.append(model(xb).detach().to('cpu').clone())
    return torch.cat(chunks, dim=0)


def export_one_flat(final_path, args, namespace, cache, model=None, split=None):
    """Convert a completed nested run to a self-contained Electricity-style .pt.

    Never reads in-progress checkpoints without metrics.json. The source files
    are preserved. Running this during an old tmux job does not interrupt it.
    """
    final = json.loads(final_path.read_text(encoding='utf-8'))
    cfg = final['config']
    if int(cfg['epochs']) != args.epochs:
        return 'filtered', None

    destination = flat_output_path(args.output_dir, cfg)
    if destination.exists():
        return 'exists', destination

    latest_path = final_path.parent / 'latest.pt'
    if not latest_path.exists():
        raise FileNotFoundError(f'Completed metrics but missing checkpoint: {latest_path}')
    saved = safe_load(latest_path)
    if saved['config'] != cfg or int(saved['epoch']) != int(cfg['epochs']):
        raise ValueError(f'Checkpoint not finished / config mismatch: {latest_path}')
    if int(final['epoch']) != int(cfg['epochs']):
        raise ValueError(f'Invalid completed epoch in {final_path}')

    notebook_sha = hashlib.sha256(args.model_notebook.read_bytes()).hexdigest()
    if cfg['model_notebook_sha256'] != notebook_sha:
        raise ValueError('Model notebook hash differs from trained checkpoint; refusing export')

    dataset = cfg['dataset']
    if dataset not in cache:
        cache[dataset] = load_bundle(args.data_dir / DATASETS[dataset])
    bundle, data_sha = cache[dataset]
    if cfg['processed_sha256'] != data_sha:
        raise ValueError('Dataset fingerprint differs from checkpoint; refusing export')

    if split is None:
        Xtr, Ytr = windows(bundle['train'], cfg['p'], cfg['h'])
        Xte, Yte = windows(bundle['test'], cfg['p'], cfg['h'])
    else:
        Xtr, Ytr, Xte, Yte = split

    temporary_model = model is None
    if temporary_model:
        namespace['create_ablation_model'].__globals__.update({
            'D_MODEL': D_MODEL, 'LSTM_HIDDEN_DIM': LSTM_HIDDEN_DIM,
            'LSTM_NUM_LAYERS': LSTM_NUM_LAYERS, 'LYAP_WEIGHT': LYAP_WEIGHT,
            'RHO_MAX': RHO_MAX, 'MODEL_VARIANTS': MODEL_VARIANTS,
        })
        model, _, _ = namespace['create_ablation_model'](
            cfg['model'], cfg['input_dim'], cfg['p'], cfg['h'])
        model.load_state_dict(saved['model_state_dict'])
        # CPU by default for a converter attached to an active GPU run.
        destination_device = args.export_device
        model = model.to(destination_device)
    else:
        destination_device = args.device

    pred_train = predict_cpu_tensor(model, Xtr, destination_device, args.eval_batch)
    pred_test = predict_cpu_tensor(model, Xte, destination_device, args.eval_batch)
    if pred_train.shape != Ytr.shape or pred_test.shape != Yte.shape:
        raise RuntimeError('Flattened checkpoint prediction/target shape mismatch')
    if not (torch.isfinite(pred_train).all() and torch.isfinite(pred_test).all()):
        raise FloatingPointError('Nonfinite predictions for standalone checkpoint')

    h = saved['history']
    if any(len(v) != cfg['epochs'] for v in h.values()):
        raise ValueError('History length mismatch in completed checkpoint')
    # Mirror Electricity checkpoint field names for downstream plotting.
    artifact = {
        'model': cfg['model'], 'dataset': cfg['dataset'],
        'model_state_dict': saved['model_state_dict'],
        'pred_train': pred_train, 'truth_train': Ytr.detach().cpu(),
        'pred_test': pred_test, 'truth_test': Yte.detach().cpu(),
        'loss': torch.tensor(h['loss'], dtype=torch.float32),
        'eig': torch.tensor(h['spectral_proxy'], dtype=torch.float32),
        'mse_history': torch.tensor(h['mse'], dtype=torch.float32),
        'lyap_history': torch.tensor(h['lyap_weighted'], dtype=torch.float32),
        'train_mse': float(final['train']['MSE']),
        'train_mae': float(final['train']['MAE']),
        'test_mse': float(final['test']['MSE']),
        'test_mae': float(final['test']['MAE']),
        'patch_len': cfg['p'], 'horizon': cfg['h'],
        'd_model': cfg['d_model'], 'lstm_hidden_dim': cfg['lstm_hidden'],
        'input_dim': cfg['input_dim'], 'epochs': cfg['epochs'],
        'learning_rate': cfg['lr'], 'lyap_weight': cfg['lyap_weight'],
        'channels': bundle.get('channels', None),
        'data_hash': cfg['processed_sha256'],
        'raw_sha256': bundle.get('raw_sha256', None),
        'config': cfg,
    }
    atomic_torch_save(artifact, destination)
    print(f'SAVED: {destination}', flush=True)
    if temporary_model:
        del model
    return 'saved', destination


def export_all_flat(args, namespace=None, cache=None):
    """Convert only finished checkpoint runs (never a partially trained one)."""
    cache = {} if cache is None else cache
    found = 0
    saved = 0
    exists = 0
    for final_path in sorted(args.output_dir.rglob('metrics.json')):
        metadata = json.loads(final_path.read_text(encoding='utf-8'))
        if int(metadata['config']['epochs']) != args.epochs:
            continue
        found += 1
        dst = flat_output_path(args.output_dir, metadata['config'])
        if dst.exists():
            exists += 1
            continue
        if namespace is None:
            namespace = load_model_notebook(args.model_notebook)
        state, _ = export_one_flat(final_path, args, namespace, cache)
        saved += (state == 'saved')
    print(f'FLAT EXPORT: {found} completed e{args.epochs} runs, '
          f'{saved} newly saved, {exists} existing', flush=True)
    return found, saved, exists


def run_one(dataset, model_name, p, h, args, namespace, cache):
    source = args.data_dir / DATASETS[dataset]
    if dataset not in cache:
        cache[dataset] = load_bundle(source)
    bundle, data_sha = cache[dataset]
    model_source_sha = hashlib.sha256(args.model_notebook.read_bytes()).hexdigest()
    cfg = {'dataset': dataset, 'model': model_name, 'p': p, 'h': h,
           'input_dim': int(bundle['input_dim']), 'epochs': args.epochs,
           'lr': args.lr, 'seed': args.seed, 'd_model': D_MODEL,
           'lstm_hidden': LSTM_HIDDEN_DIM, 'lstm_layers': LSTM_NUM_LAYERS,
           'rho_max': RHO_MAX,
           'lyap_weight': LYAP_WEIGHT if model_name == 'KoopPatchTST_Full' else 0.,
           'processed_sha256': data_sha, 'model_notebook_sha256': model_source_sha}
    folder = args.output_dir / output_id(dataset, model_name, p, h, args.epochs, args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    # A restart creates a NEW run; do not reuse a previous flat final artifact.
    if args.restart:
        old_flat = flat_output_path(args.output_dir, cfg)
        if old_flat.exists():
            old_flat.unlink()
    final_path = folder / 'metrics.json'
    latest_path = folder / 'latest.pt'
    if final_path.exists() and not args.restart:
        final = json.loads(final_path.read_text(encoding='utf-8'))
        if final.get('config') != cfg:
            raise ValueError(f'Finished config mismatch: {final_path}')
        flat_file = flat_output_path(args.output_dir, cfg)
        if not flat_file.exists():
            export_one_flat(final_path, args, namespace, cache)
        print(f'SKIP COMPLETED {dataset} {model_name} P={p} H={h}', flush=True)
        return final
    Xtr, Ytr = windows(bundle['train'], p, h)
    Xte, Yte = windows(bundle['test'], p, h)
    # The author uses full-batch training, including ALL training windows.
    Xg, Yg = Xtr.to(args.device), Ytr.to(args.device)
    namespace['create_ablation_model'].__globals__.update({
        'D_MODEL': D_MODEL, 'LSTM_HIDDEN_DIM': LSTM_HIDDEN_DIM,
        'LSTM_NUM_LAYERS': LSTM_NUM_LAYERS, 'LYAP_WEIGHT': LYAP_WEIGHT,
        'RHO_MAX': RHO_MAX, 'MODEL_VARIANTS': MODEL_VARIANTS,
    })
    seed_everything(args.seed)
    model, koop_attr, lyap_w = namespace['create_ablation_model'](
        model_name, cfg['input_dim'], p, h)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch_start = 0
    history = {'loss': [], 'mse': [], 'lyap_weighted': [], 'spectral_proxy': []}
    if latest_path.exists() and not args.restart:
        ckpt = safe_load(latest_path)
        if ckpt.get('config') != cfg:
            raise ValueError(f'Checkpoint config mismatch: {latest_path}')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Optimizer states were saved on CPU and must follow the model to GPU.
        for state in optimizer.state.values():
            for key, val in state.items():
                if torch.is_tensor(val):
                    state[key] = val.to(args.device)
        epoch_start = int(ckpt['epoch'])
        history = ckpt['history']
        restore_rng(ckpt['rng'])
        print(f'RESUME {dataset} {model_name} P={p} H={h} at epoch {epoch_start}', flush=True)
    elif args.restart and final_path.exists():
        final_path.unlink()
    if any(len(v) != epoch_start for v in history.values()):
        raise ValueError('Checkpoint history length mismatch')
    start_wall = time.perf_counter()
    for ep in range(epoch_start, args.epochs):
        model.train()
        optimizer.zero_grad()
        total, mse, lyap = loss_parts(model, Xg, Yg, lyap_w)
        if not torch.isfinite(total).item():
            raise FloatingPointError(f'Non-finite loss at epoch {ep}')
        total.backward()
        optimizer.step()
        history['loss'].append(float(total.detach().item()))
        history['mse'].append(float(mse.detach().item()))
        history['lyap_weighted'].append(float((lyap_w*lyap).detach().item()))
        koop = getattr(model, koop_attr, None) if koop_attr else None
        spectral = float(torch.sigmoid(koop.S_raw).detach().abs().max().item()) if isinstance(
            koop, namespace['StrictStableKoopmanOperator']) else 0.
        history['spectral_proxy'].append(spectral)
        if ep % max(1, args.epochs // 10) == 0 or ep+1 == args.epochs:
            print(f'  {dataset:15} {model_name:25} p={p} h={h} '
                  f'epoch={ep+1}/{args.epochs} mse={mse.item():.6g} '
                  f'lyap={(lyap_w*lyap).item():.4g}', flush=True)
        if (ep+1) % args.checkpoint_every == 0 or ep+1 == args.epochs:
            # Checkpoints include optimizer and PRNG states for exact-on-host resumes.
            snap = {'config': cfg, 'epoch': ep+1,
                    'model_state_dict': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history, 'rng': snapshot_rng()}
            atomic_torch_save(snap, latest_path)
    if args.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_wall
    train_scores = metrics(model, Xtr, Ytr, args.device, args.eval_batch)
    test_scores = metrics(model, Xte, Yte, args.device, args.eval_batch)
    final = {'config': cfg, 'epoch': args.epochs, 'train': train_scores,
             'test': test_scores, 'elapsed_seconds_this_session': elapsed,
             'final_train_loss': history['loss'][-1]}
    atomic_json_save(final, final_path)
    with open(folder / 'training_history.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['epoch'] + list(history))
        for i in range(len(history['loss'])):
            w.writerow([i+1] + [history[k][i] for k in history])
    export_one_flat(final_path, args, namespace, cache, model=model,
                    split=(Xtr, Ytr, Xte, Yte))
    print(f'DONE {dataset} {model_name} P={p} H={h} '
          f'Test MSE={test_scores["MSE"]:.6g} MAE={test_scores["MAE"]:.6g} '
          f'time={elapsed:.1f}s', flush=True)
    del model, optimizer, Xtr, Ytr, Xte, Yte, Xg, Yg
    gc.collect()
    if args.device.type == 'cuda':
        torch.cuda.empty_cache()
    return final


def export_summary(output_dir, epochs=None):
    rows = []
    for path in output_dir.rglob('metrics.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            cfg = data['config']
            if epochs is not None and int(cfg['epochs']) != epochs:
                continue
            for split in ('train', 'test'):
                row = {'Dataset': cfg['dataset'], 'Model': cfg['model'],
                       'PatchLen': cfg['p'], 'Horizon': cfg['h'],
                       'Epochs': cfg['epochs'], 'Seed': cfg['seed'],
                       'Set': split.title(), 'MSE': data[split]['MSE'],
                       'MAE': data[split]['MAE'], 'Samples': data[split]['Samples']}
                rows.append(row)
        except (KeyError, ValueError) as exc:
            print(f'WARNING: cannot aggregate {path}: {exc}', file=sys.stderr)
    rows.sort(key=lambda x:(x['Dataset'],x['PatchLen'],x['Horizon'],x['Model'],x['Set']))
    if rows:
        target = output_dir / 'metrics.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.csv.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, target)
    print(f'Completed configurations: {len(rows)//2}; metrics rows: {len(rows)}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-notebook', type=Path, default=Path(__file__).resolve().parent/'02_ablation_model.ipynb')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results'/'ablation_runs')
    parser.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=list(MODEL_VARIANTS), default=list(MODEL_VARIANTS))
    parser.add_argument('--patches', nargs='+', type=int, default=list(PATCH_LENGTHS))
    parser.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--eval-batch', type=int, default=128)
    parser.add_argument('--limit', type=int, default=None, help='Run at most N configurations (for smoke tests)')
    parser.add_argument('--restart', action='store_true', help='Ignore existing checkpoint/results for selected configurations')
    parser.add_argument('--run', action='store_true', help='Perform training; without this flag only list the grid')
    parser.add_argument('--summary-only', action='store_true', help='Collect existing metrics without loading models/data')
    parser.add_argument('--export-flat', action='store_true', help='Convert completed nested checkpoint runs to Electricity-style flat .pt files; no training')
    parser.add_argument('--watch-flat', action='store_true', help='Continuously export completed runs while another tmux job trains')
    parser.add_argument('--flat-interval', type=int, default=30, help='Seconds between flat exporter scans (default 30)')
    parser.add_argument('--export-device', choices=('cpu','cuda'), default='cpu', help='Device for old-checkpoint conversion, default CPU to avoid contending with GPU training')
    args = parser.parse_args()
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but not available')
    for value in args.patches:
        if value not in PATCH_LENGTHS:
            parser.error(f'Unsupported patch length {value}')
    for value in args.horizons:
        if value not in HORIZONS:
            parser.error(f'Unsupported horizon {value}')
    if args.epochs < 1 or args.checkpoint_every < 1 or args.eval_batch < 1 or args.limit is not None and args.limit < 1 or args.flat_interval < 1:
        parser.error('Epochs, checkpoint interval, eval batch and limit must be positive')
    return args


def main():
    args = parse_args()
    if args.summary_only:
        export_summary(args.output_dir, args.epochs)
        return
    if args.export_flat or args.watch_flat:
        if args.export_device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA not available for converter')
        # The exporter can run without affecting an ongoing training process.
        if args.export_device == 'cpu':
            torch.set_num_threads(min(2, torch.get_num_threads()))
        cache = {}
        while True:
            export_all_flat(args, cache=cache)
            if not args.watch_flat:
                return
            time.sleep(args.flat_interval)
    configs = list(itertools.product(args.datasets, args.patches, args.horizons, args.models))
    if args.limit is not None:
        configs = configs[:args.limit]
    print(f'GPU: {torch.cuda.get_device_name(0) if args.device.type == "cuda" else "CPU"}', flush=True)
    print(f'Grid: {len(configs)} configurations; epochs: {args.epochs}', flush=True)
    print(f'Model notebook: {args.model_notebook}', flush=True)
    print(f'Data directory: {args.data_dir}', flush=True)
    print(f'Result directory: {args.output_dir}', flush=True)
    if not args.run:
        print('DRY RUN ONLY. Add --run to start training.', flush=True)
        for ds, p, h, mod in configs[:20]:
            print(f'  {ds:15} P={p} H={h} {mod}')
        if len(configs)>20:
            print(f'  ... {len(configs)-20} more')
        return
    namespace = load_model_notebook(args.model_notebook)
    print(f'Model definitions loaded: {len(DEFINITIONS)}', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for i, (ds, p, h, mod) in enumerate(configs, 1):
        print(f'[{i}/{len(configs)}] {ds} {mod} P={p} H={h}', flush=True)
        try:
            run_one(ds, mod, p, h, args, namespace, cache)
        except Exception:
            traceback.print_exc()
            export_summary(args.output_dir, args.epochs)
            print('Stopping at first failure; checkpoint remains available for resume.', file=sys.stderr)
            raise
        export_summary(args.output_dir, args.epochs)
    print('ALL REQUESTED CONFIGURATIONS FINISHED', flush=True)


if __name__ == '__main__':
    main()
