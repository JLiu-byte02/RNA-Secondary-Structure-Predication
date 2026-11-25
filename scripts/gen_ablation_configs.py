#!/usr/bin/env python
"""Generate the paper-ablation configs from frozen base recipes.

Two families of configs are produced, all from existing, already-validated YAML
so that data splits, decoder, optimiser, unfreeze schedule and selection metric
are bit-identical to the primary runs:

TR0 (bpRNA standard split, base = ``configs/exp010_tr0_bpp.yaml``)
  tr0_thermo_s{17,1024,2026}   full method, three seeds
  tr0_seq_s{17,1024,2026}      matched no-BPP control (wide2d head, no BPP)
  tr0_zerobpp_s17              same parameters as the method, BPP channel zeroed
  tr0_deepseq_s17              no-BPP, 6-block outer-product head (capacity control)
  tr0_nos0_s17                 wide head without the base-logit channel
  tr0_nodist_s17               wide head without the distance channel
  tr0_nomask_s17               wide head without the mask channel
  tr0_bpponly_s17              wide head seeing only BPP
  tr0_mfe_s17                  BPP channel replaced by binary MFE pairs
  tr0_multithermo_s17          [bpp, mfe, mea] concatenated

ArchiveII folds (base = ``configs/folds/{method_bpp,control}_<fam>.yaml``)
  zerobpp_<fam>                zero-BPP control on 5s / srp
  deepseq_<fam>                6-block no-BPP control on 5s / srp
  method_bpp_5s_s{1024,2026}   extra seeds on the 5s fold (P1 multi-seed)

Usage:
  python scripts/gen_ablation_configs.py [--out configs/ablations]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def save(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def force_no_flash(cfg: dict) -> None:
    """Use the pure-PyTorch attention path.

    The released RiNALMo weights in this workspace are the native (non-flash)
    parameterisation; the flash path packs q/k/v into a fused ``Wqkv`` tensor and
    would not load them.  Both paths compute the same function; only the kernel
    and state-dict layout differ.
    """
    cfg.setdefault("model", {}).setdefault("backbone", {})["flash_attn"] = False


def set_thermo(cfg: dict, data_channels: list[str]) -> None:
    """Set data/model channel declarations consistently."""
    cfg.setdefault("data", {})["channels"] = list(data_channels)
    cfg["data"]["thermo_dir"] = "data/thermo/all" if data_channels else None
    cfg.setdefault("model", {})["channels"] = list(data_channels)


def set_wide_head(
    cfg: dict,
    head_channels: list[str],
    *,
    zero_channels: list[str] | None = None,
) -> None:
    cfg["model"]["head"] = {
        "type": "wide2d",
        "kwargs": {
            "base_blocks": 2,
            "channels": list(head_channels),
            "conv_dim": 64,
            "depth": 4,
            "kernel_size": 3,
            "zero_init": True,
            **({"zero_channels": list(zero_channels)} if zero_channels else {}),
        },
    }


def set_deep_outer_head(cfg: dict) -> None:
    cfg["model"]["head"] = {
        "type": "outer_product",
        "kwargs": {"num_blocks": 6, "conv_dim": 64, "kernel_size": 3},
    }


def tr0_configs() -> dict[str, dict]:
    base = load("configs/exp010_tr0_bpp.yaml")
    out: dict[str, dict] = {}

    def variant(name: str, seed: int, data_channels, head_channels, **kwargs):
        cfg = copy.deepcopy(base)
        cfg["run_name"] = name
        cfg["seed"] = int(seed)
        set_thermo(cfg, data_channels)
        set_wide_head(cfg, head_channels, **kwargs)
        out[name] = cfg

    for seed in (17, 1024, 2026):
        cfg = copy.deepcopy(base)
        cfg["run_name"] = f"tr0_thermo_s{seed}"
        cfg["seed"] = seed
        set_thermo(cfg, ["bpp"])
        set_wide_head(cfg, ["s0", "bpp", "distance", "mask"])
        out[cfg["run_name"]] = cfg

        cfg = copy.deepcopy(base)
        cfg["run_name"] = f"tr0_seq_s{seed}"
        cfg["seed"] = seed
        set_thermo(cfg, [])
        set_wide_head(cfg, ["s0", "distance", "mask"])
        out[cfg["run_name"]] = cfg

    variant(
        "tr0_zerobpp_s17",
        17,
        ["bpp"],
        ["s0", "bpp", "distance", "mask"],
        zero_channels=["bpp"],
    )
    cfg = copy.deepcopy(base)
    cfg["run_name"] = "tr0_deepseq_s17"
    cfg["seed"] = 17
    set_thermo(cfg, [])
    set_deep_outer_head(cfg)
    out[cfg["run_name"]] = cfg

    # The checklist defines "Seq-Only" as backbone + the official 2-block
    # outer-product head (no wide2d at all).  Keep this separate from
    # tr0_seq_s17 (wide2d without BPP) so head-upgrade and BPP effects can be
    # decomposed instead of confounded.
    cfg = copy.deepcopy(base)
    cfg["run_name"] = "tr0_officialhead_s17"
    cfg["seed"] = 17
    set_thermo(cfg, [])
    cfg["model"]["head"] = {
        "type": "outer_product",
        "kwargs": {"num_blocks": 2, "conv_dim": 64, "kernel_size": 3},
    }
    out[cfg["run_name"]] = cfg

    variant("tr0_nos0_s17", 17, ["bpp"], ["bpp", "distance", "mask"])
    variant("tr0_nodist_s17", 17, ["bpp"], ["s0", "bpp", "mask"])
    variant("tr0_nomask_s17", 17, ["bpp"], ["s0", "bpp", "distance"])
    variant("tr0_bpponly_s17", 17, ["bpp"], ["bpp"])
    variant("tr0_mfe_s17", 17, ["mfe"], ["s0", "mfe", "distance", "mask"])
    variant(
        "tr0_multithermo_s17",
        17,
        ["bpp", "mfe", "mea"],
        ["s0", "bpp", "mfe", "mea", "distance", "mask"],
    )
    return out


def archiveii_configs() -> dict[str, dict]:
    out: dict[str, dict] = {}

    # Re-train the six matched no-BPP controls that already exist in the local
    # results (5s, srp, RNaseP, tRNA, telomerase, tmRNA).  The old server run
    # directory was removed, so the per-epoch checkpoints needed for the
    # single-best / soup 2x2 are gone; re-running the identical config is the
    # only way to get a matched soup cell.  Run names get a "_v2" suffix.
    for family in ("5s", "srp", "RNaseP", "tRNA", "telomerase", "tmRNA"):
        cfg = load(f"configs/folds/control_{family}.yaml")
        cfg["run_name"] = f"control_{family}_v2"
        out[cfg["run_name"]] = cfg

    for family in ("5s", "srp"):
        method = load(f"configs/folds/method_bpp_{family}.yaml")
        control = load(f"configs/folds/control_{family}.yaml")

        cfg = copy.deepcopy(method)
        cfg["run_name"] = f"zerobpp_{family}"
        cfg["model"]["head"]["kwargs"]["zero_channels"] = ["bpp"]
        out[cfg["run_name"]] = cfg

        cfg = copy.deepcopy(control)
        cfg["run_name"] = f"deepseq_{family}"
        set_deep_outer_head(cfg)
        out[cfg["run_name"]] = cfg

    cfg = load("configs/folds/method_bpp_5s.yaml")
    cfg["run_name"] = "method_bpp_5s_v2"
    out[cfg["run_name"]] = cfg
    for seed in (1024, 2026):
        cfg = load("configs/folds/method_bpp_5s.yaml")
        cfg["run_name"] = f"method_bpp_5s_s{seed}"
        cfg["seed"] = seed
        out[cfg["run_name"]] = cfg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/ablations")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    configs = {**tr0_configs(), **archiveii_configs()}
    for name, cfg in configs.items():
        force_no_flash(cfg)
        path = out_dir / f"{name}.yaml"
        save(cfg, path)
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"wrote {shown}")
    print(f"{len(configs)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
