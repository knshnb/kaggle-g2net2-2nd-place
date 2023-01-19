from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cupy as xp
import h5py
import numpy as np
import pandas as pd
import pyfstat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.config import Config, load_config


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched filter for Kaggle g2net2")
    parser.add_argument("--config_path", type=str, default="config/debug.yaml")
    parser.add_argument("--out_dir", type=str, default="result/tmp")
    parser.add_argument("--in_base_dir", type=str, default="input")
    parser.add_argument("--data_name", type=str, default="train")
    parser.add_argument("--n_data", type=int, default=-1)
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--topk", type=int, default=100)
    return parser.parse_args()


def read_comp_data(data_path, scale: float = 1e22) -> dict[str, np.ndarray]:
    with h5py.File(data_path, "r") as f:
        filename = Path(data_path).name.replace(".hdf5", "")
        f = f[filename]
        h1 = f["H1"]
        l1 = f["L1"]
        return {
            "H1": h1["SFTs"][()] * scale,
            "H1_ts": h1["timestamps_GPS"][()],
            "L1": l1["SFTs"][()] * scale,
            "L1_ts": l1["timestamps_GPS"][()],
            "freq_hz": f["frequency_Hz"][()],
        }


def preprocess_real_data(
    stft_sq: np.ndarray, median_coeff: float = 1.1, percent: int = 75, normalize_width: int = 100
) -> np.ndarray:
    # 1. Mask frequencies with high std
    freq_std = np.std(stft_sq, axis=1)
    error_freq = (freq_std > np.median(freq_std) * median_coeff) & (freq_std > np.percentile(freq_std, percent))
    stft_sq[error_freq] = stft_sq[~error_freq].mean(axis=0)

    # 2. Timewise normalize
    _, n_time = stft_sq.shape
    ret = np.empty_like(stft_sq)
    for i in range(n_time):
        s, t = max(i - normalize_width, 0), min(i + normalize_width + 1, n_time)
        std = stft_sq[:, s:t].mean()
        ret[:, i] = stft_sq[:, i] * (1.5 * 1.5 / std)
    return ret


def calc_score_batch(
    timestamp: xp.ndarray,  # (L)
    velocities: xp.ndarray,  # (3, L)
    stft_sq: xp.ndarray,  # (360, L)
    freq_ref: float,
    freq_dif: float,
    n: xp.ndarray,  # (n_batch, 3)
    F0: xp.ndarray,  # (n_batch)
    F1: xp.ndarray,  # (n_batch)
    tref: xp.ndarray,  # (n_batch)
    freq_width: int,
) -> xp.ndarray:  # (n_batch)
    """Calculate average of the powers corresponding to the signal (by batch)."""
    n_freq = len(stft_sq)
    f_h = (F0[:, None] + (timestamp[None, :] - tref[:, None]) * F1[:, None]) * (1 + xp.dot(n, velocities))
    # (n_batch, L)

    orig_signal_idx_float = (f_h - freq_ref) / freq_dif
    signal_idx_float = orig_signal_idx_float.clip(0.0, n_freq - 1)
    # (n_batch, L)
    range_arr = xp.arange(len(timestamp))
    half_freq_width = freq_width // 2
    if freq_width == 2:
        # Adhoc speedup
        floor_idx = xp.floor(signal_idx_float).astype(int)
        dif = xp.abs(floor_idx - signal_idx_float)
        signal_part = stft_sq[floor_idx, range_arr] * (1 - dif) + stft_sq[floor_idx + 1, range_arr] * dif
    else:
        if freq_width % 2 == 0:
            floor_idx = xp.floor(signal_idx_float).astype(int)[:, None, :]
            idxs = xp.concatenate([floor_idx + i for i in range(-half_freq_width + 1, half_freq_width + 1)], axis=1)
        else:
            round_idx = xp.round(signal_idx_float).astype(int)[:, None, :]
            idxs = xp.concatenate([round_idx + i for i in range(-half_freq_width, half_freq_width + 1)], axis=1)
            # (n_batch, freq_width, L)
        idxs = idxs.clip(min=0)
        dif = xp.abs(idxs - signal_idx_float[:, None, :])
        weights = 1.0 / (dif + 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        signal_part = (stft_sq[idxs, range_arr] * weights).sum(axis=1)
        # (n_batch, L)

    score = xp.sqrt(signal_part.mean(axis=1))
    score[(orig_signal_idx_float.min(axis=1) < -1) | (orig_signal_idx_float.max(axis=1) >= n_freq + 1)] = -1.0
    return score


def sample_params(n_sample: int, F0_l: float, F0_r: float) -> dict[str, xp.ndarray]:
    n_neg = round(n_sample * 2 / 3)
    n_pos = n_sample - n_neg
    F1_neg = -2 * xp.power(10.0, xp.random.uniform(-11, -8, n_neg))
    F1_pos = 2 * xp.power(10.0, xp.random.uniform(-11, -9, n_pos))
    return {
        "alpha": xp.random.uniform(0.0, 2 * xp.pi, n_sample),
        "delta": xp.arcsin(xp.random.uniform(-1.0, 1.0, n_sample)),
        "F0": xp.random.beta(2, 2, n_sample) * (F0_r - F0_l) + F0_l,
        "F1": xp.concatenate([F1_neg, F1_pos]),
    }


def get_velocity(timestamp: np.ndarray, detector_name: str) -> np.ndarray:
    detector_states = pyfstat.DetectorStates()
    states = detector_states.get_multi_detector_states(timestamp, 1800, detector_name)
    return np.vstack([data.vDetector for data in states.data[0].data]).T


def random_search(data: dict[str, np.ndarray], cfg: Config, is_real: bool) -> dict[str, np.ndarray]:
    """Random search of alpha, delta, F0, and F1."""
    h1_ts = xp.asarray(data["H1_ts"])
    velocity_h = xp.asarray(get_velocity(data["H1_ts"], "H1"))
    stft_h1 = xp.asarray(data["H1"].real ** 2 + data["H1"].imag ** 2)
    l1_ts = xp.asarray(data["L1_ts"])
    velocity_l = xp.asarray(get_velocity(data["L1_ts"], "L1"))
    stft_l1 = xp.asarray(data["L1"].real ** 2 + data["L1"].imag ** 2)
    tref = xp.broadcast_to(xp.array([1238170000]), (cfg.n_batch,))

    if is_real:
        stft_h1 = preprocess_real_data(stft_h1)
        stft_l1 = preprocess_real_data(stft_l1)

    freq_hz = data["freq_hz"]
    F0_l = freq_hz[0] * 1.2 + freq_hz[-1] * -0.2
    F0_r = freq_hz[0] * -0.2 + freq_hz[-1] * 1.2
    score_list: dict[str, list[np.ndarray]] = defaultdict(list)
    while sum(len(x) for x in score_list["score"]) < cfg.n_trial:
        params = sample_params(cfg.n_batch, F0_l, F0_r)
        alpha, delta = params["alpha"], params["delta"]
        n = xp.stack([xp.cos(alpha) * xp.cos(delta), xp.sin(alpha) * xp.cos(delta), xp.sin(delta)], axis=1)
        freq_dif = freq_hz[1] - freq_hz[0]
        F0, F1 = params["F0"], params["F1"]
        score_h1 = calc_score_batch(h1_ts, velocity_h, stft_h1, freq_hz[0], freq_dif, n, F0, F1, tref, cfg.freq_width)
        score_l1 = calc_score_batch(l1_ts, velocity_l, stft_l1, freq_hz[0], freq_dif, n, F0, F1, tref, cfg.freq_width)
        score = (score_h1 + score_l1) / 2
        feasible_mask = ((score_h1 != -1.0) & (score_l1 != -1.0)).get()
        score_list["score"].append(score.get()[feasible_mask])
        score_list["score_h1"].append(score_h1.get()[feasible_mask])
        score_list["score_l1"].append(score_l1.get()[feasible_mask])
        for key, val in params.items():
            score_list[key].append(val.get()[feasible_mask])
    return {key: np.concatenate(val)[: cfg.n_trial] for key, val in score_list.items()}


def predict(args: argparse.Namespace):
    cfg = load_config(args.config_path, "config/default.yaml")
    if args.data_name == "train":
        df = pd.read_csv(f"{args.in_base_dir}/train_labels.csv")
        df["is_real"] = False
    elif args.data_name == "test":
        df = pd.read_csv(f"{args.in_base_dir}/sample_submission.csv")
        df["is_real"] = df.id.isin(pd.read_csv(f"{args.in_base_dir}/test_real.csv").id)
    else:
        raise ValueError
    # https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/363734
    df = df[df.target != -1]
    if args.n_data != -1:
        df, _ = train_test_split(df, train_size=args.n_data, random_state=0, stratify=df.target)

    n_each = (len(df) + args.n_parallel - 1) // args.n_parallel
    df = df[n_each * args.parallel_idx : n_each * (args.parallel_idx + 1)]
    print(df)

    preds = []
    df_score_list = []
    for row in tqdm(df.itertuples()):
        data = read_comp_data(f"{args.in_base_dir}/{args.data_name}/{row.id}.hdf5")
        xp.random.seed(args.seed)
        result = random_search(data, cfg, is_real=row.is_real)
        top_idx = np.argpartition(-result["score"], args.topk)[: args.topk]
        top_idx = top_idx[(-result["score"][top_idx]).argsort()]
        df_score = pd.DataFrame({key: val[top_idx] for key, val in result.items()})
        if args.verbose:
            print(df_score)
        pred = df_score.score.max()
        # Generated data with high scores are considered to be positive.
        if not row.is_real and pred > cfg.positive_threshold:
            pred += 1.0
        preds.append(pred)
        df_score["id"] = row.id
        df_score_list.append(df_score)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(df_score_list, ignore_index=True).to_csv(out_dir / "score.csv", index=False)
    pd.DataFrame({"id": df.id, "target": preds}).to_csv(out_dir / "pred.csv", index=False)
    if args.data_name == "train":
        print("AUC:", roc_auc_score(df.target, preds))


if __name__ == "__main__":
    pyfstat.set_up_logger(log_level="WARNING")
    predict(parse())
