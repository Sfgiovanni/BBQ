# -*- coding: utf-8 -*-
"""
Driver do experimento BBQ v2: gera dados, roda scoring, calcula métricas e
salva tudo em results/run_<timestamp>_<model_tag>/.

Uso: python run_experiment.py <phase>  (phase ∈ {smoke, pilot, full})
Ou importado pelo notebook BBQ_v2.ipynb via run_phase(...).
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import bbq_v2_lib as lib
import metrics as M

PHASES = {
    # n_logical: nº de exemplos lógicos (cada um vira 3 permutações)
    "smoke": {"n_logical": 30, "audit_n": 12, "with_ci": False, "n_boot": 0},
    "pilot": {"n_logical": 600, "audit_n": 200, "with_ci": True, "n_boot": 500},
    "full": {"n_logical": None, "audit_n": 200, "with_ci": True, "n_boot": 500},
}


def make_run_dir(phase):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_tag = lib.MODEL_NAME.split("/")[-1]
    run_dir = Path("results") / f"run_{ts}_{model_tag}_{phase}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def build_data(phase_cfg):
    df_templates = lib.build_base_templates()
    df_logical = lib.build_logical_examples(df_templates)
    n = phase_cfg["n_logical"]
    if n is not None:
        df_logical = lib.sample_balanced_logical(df_logical, n)
    df_expanded = lib.expand_permutations(df_logical)
    return df_templates, df_logical, df_expanded


def compute_and_save_metrics(df, run_dir, with_ci=True, n_boot=500):
    M.metrics_table(df, [], with_ci=with_ci, n_boot=n_boot).to_csv(run_dir / "metrics_overall.csv", index=False)
    M.metrics_table(df, ["category"], with_ci=with_ci, n_boot=n_boot).to_csv(run_dir / "metrics_by_category.csv", index=False)
    M.metrics_table(df, ["category", "bias_target_label"]).to_csv(run_dir / "metrics_by_label.csv", index=False)
    M.unknown_by_position(df).to_csv(run_dir / "metrics_by_position.csv", index=False)
    M.metrics_table(df, ["scenario_name"]).to_csv(run_dir / "metrics_by_scenario.csv", index=False)
    M.metrics_table(df, ["context_type", "question_type"]).to_csv(run_dir / "metrics_by_condition.csv", index=False)


def run_phase(phase, model=None, tokenizer=None, batch_size=8):
    cfg = PHASES[phase]
    torch.manual_seed(lib.RANDOM_SEED)
    np.random.seed(lib.RANDOM_SEED)

    run_dir = make_run_dir(phase)
    df_templates, df_logical, df_expanded = build_data(cfg)
    df_templates.to_csv(run_dir / "base_templates.csv", index=False, encoding="utf-8")
    df_expanded.to_parquet(run_dir / "expanded_examples.parquet", index=False)

    config = {
        "phase": phase,
        "model_name": lib.MODEL_NAME,
        "dtype": lib.DTYPE,
        "seed": lib.RANDOM_SEED,
        "eval_method": "logprob_first_token",
        "prompt_fix": "unknown por conteudo, sem letra fixa",
        "n_logical_examples": int(len(df_logical)),
        "n_expanded_examples": int(len(df_expanded)),
        "n_permutations": 3,
        "batch_size": batch_size,
        "audit_n": cfg["audit_n"],
        "max_new_tokens_audit": lib.MAX_NEW_TOKENS_AUDIT,
        "do_sample": False,
        "timestamp": datetime.now().isoformat(),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if model is None:
        tokenizer, model = lib.load_model()

    t0 = time.time()
    df_results = lib.run_scoring(
        df_expanded, model, tokenizer, run_dir,
        batch_size=batch_size, audit_n=cfg["audit_n"],
    )
    elapsed = time.time() - t0

    usage = lib.summarize_tokens(df_results, run_dir)
    agreement, df_audit = lib.audit_agreement(df_results)
    if agreement is not None:
        cols = ["logical_id", "perm_index", "unknown_position", "correct_option",
                "predicted_option", "parsed_generation_option", "raw_generation",
                "logprob_A", "logprob_B", "logprob_C"]
        df_audit[cols].to_csv(run_dir / "audit_generation_vs_logprob.csv", index=False, encoding="utf-8")
        with open(run_dir / "audit_agreement.json", "w", encoding="utf-8") as f:
            json.dump(agreement, f, ensure_ascii=False, indent=2)

    compute_and_save_metrics(df_results, run_dir, with_ci=cfg["with_ci"], n_boot=cfg["n_boot"])

    print(f"\n[{phase}] concluído em {elapsed/60:.1f} min — resultados em {run_dir}")
    print("Tokens:", usage)
    if agreement:
        print("Auditoria geração vs logprob:", agreement)
    return run_dir, df_results, model, tokenizer


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    run_phase(phase)
