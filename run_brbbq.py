# -*- coding: utf-8 -*-
"""
Driver da rodada BR-BBQ (BR-específico).

Reusa sem modificação as primitivas de inferência de bbq_v2_lib (build_prompt,
score_options, generate_audit, letter_token_ids, parse_model_answer), as
métricas de metrics.py e o relatório de make_report.py. A única lógica nova é:

  1. fonte de dados = brbbq_data (JSONL BR, em vez dos templates gerados);
  2. loop de scoring com checkpoint a cada 20 exemplos + retomada;
  3. geração livre de auditoria em TODOS os exemplos (dataset pequeno);
  4. heartbeat em PROGRESS.md a cada 15 min / checkpoint / transição de etapa.

Uso:
    python run_brbbq.py            # rodada completa (216 forwards)
    python run_brbbq.py smoke      # smoke test (10 lógicos × 3 = 30 forwards)
"""

import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import bbq_v2_lib as lib
import brbbq_data as data
import metrics as M
import make_report

HEARTBEAT_SECONDS = 900          # 15 min
CHECKPOINT_EVERY = 20            # exemplos (linhas expandidas)
SMOKE_N_LOGICAL = 10
BATCH_SIZE = 4


# ============================================================
# PROGRESS.md — heartbeat
# ============================================================

class ProgressLog:
    """Escreve entradas (append) em PROGRESS.md a cada 15 min, checkpoint e
    transição de etapa, para que uma interrupção deixe claro onde parou."""

    def __init__(self, path, total_forwards, config):
        self.path = Path(path)
        self.total = total_forwards
        self.t_start = time.monotonic()
        self.t_last_log = self.t_start
        header = [
            "# PROGRESS — rodada BR-BBQ\n",
            f"Início: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Config: modelo=`{config['model_name']}`, dtype={config['dtype']}, "
            f"seed={config['seed']}, exemplos_lógicos={config['n_logical_examples']}, "
            f"forwards_alvo={self.total}, batch={config['batch_size']}, "
            f"método={config['eval_method']}.",
            "",
        ]
        self.path.write_text("\n".join(header), encoding="utf-8")

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def write(self, stage, done, tok_in, tok_out, last_cats, note=""):
        elapsed = time.monotonic() - self.t_start
        pct = 100.0 * done / self.total if self.total else 0.0
        per_ex = elapsed / done if done else float("nan")
        remaining = self.total - done
        eta = (datetime.now() + timedelta(seconds=per_ex * remaining)) if done else None
        eta_s = eta.strftime("%H:%M") if eta else "—"
        lat = f"{per_ex:.2f}" if done else "—"
        entry = [
            f"## [{self._now()}]",
            f"- Etapa: {stage}",
            f"- Progresso: {done}/{self.total} forwards ({pct:.0f}%)",
            f"- Latência média: {lat} s/exemplo | ETA: {eta_s}",
            f"- Tokens acumulados: entrada {tok_in:,}, saída {tok_out:,}",
            f"- Últimas categorias processadas: {last_cats or '—'}",
            f"- Problemas/observações: {note or 'nenhum'}",
            "",
        ]
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(entry))
        self.t_last_log = time.monotonic()

    def maybe_heartbeat(self, *args, **kwargs):
        if time.monotonic() - self.t_last_log >= HEARTBEAT_SECONDS:
            self.write(*args, **kwargs)


# ============================================================
# Loop de scoring com checkpoint / retomada / auditoria total
# ============================================================

def run_scoring_br(df_expanded, model, tokenizer, run_dir, plog,
                   batch_size=BATCH_SIZE, stage="rodada completa"):
    """Scoring por logprobs + geração de auditoria em TODOS os exemplos, com
    checkpoint (csv+parquet) a cada CHECKPOINT_EVERY e retomada por example_id."""
    run_dir = Path(run_dir)
    ckpt_stem = run_dir / "raw_predictions"
    ckpt_parquet = Path(str(ckpt_stem) + ".parquet")

    if ckpt_parquet.exists():
        df_done = pd.read_parquet(ckpt_parquet)
        done_ids = set(df_done["example_id"].astype(str))
        rows = df_done.to_dict("records")
        print(f"Retomando: {len(done_ids)} exemplos já salvos.")
        plog.write(stage, len(rows),
                   int(df_done["n_input_tokens"].sum()),
                   int(df_done["n_output_tokens"].sum()),
                   "(retomada)", note=f"retomando de checkpoint com {len(rows)} linhas")
    else:
        done_ids, rows = set(), []

    letter_ids = lib.letter_token_ids(tokenizer)
    df_todo = df_expanded[~df_expanded["example_id"].astype(str).isin(done_ids)]
    records = df_todo.to_dict("records")

    tok_in = sum(int(r.get("n_input_tokens", 0)) for r in rows)
    tok_out = sum(int(r.get("n_output_tokens", 0)) for r in rows)
    last_cats = []
    n_since_save = 0

    from tqdm.auto import tqdm
    n_batches = math.ceil(len(records) / batch_size) if records else 0
    for start in tqdm(range(0, len(records), batch_size), total=n_batches):
        batch = records[start:start + batch_size]
        prompts = [lib.build_prompt(ex) for ex in batch]

        t0 = time.time()
        scored = lib.score_options(model, tokenizer, prompts, letter_ids)
        batch_latency = (time.time() - t0) / len(batch)

        for ex, sc in zip(batch, scored):
            row = dict(ex)
            row.update(sc)
            row["model_name"] = lib.MODEL_NAME
            row["latency_s"] = batch_latency
            row["is_correct"] = row["predicted_option"] == row["correct_option"]
            row["is_biased_answer"] = row["predicted_option"] == row["biased_option"]
            row["is_valid"] = True

            # geração livre de auditoria em TODOS os exemplos (dataset pequeno)
            gen_text, n_gen = lib.generate_audit(model, tokenizer, lib.build_prompt(ex))
            row["raw_generation"] = gen_text
            row["n_output_tokens"] = int(n_gen)
            row["parsed_generation_option"] = lib.parse_model_answer(
                gen_text, ex["option_A"], ex["option_B"], ex["option_C"])

            rows.append(row)
            tok_in += int(row["n_input_tokens"])
            tok_out += int(row["n_output_tokens"])
            last_cats.append(row["category_code"])
            n_since_save += 1

        if n_since_save >= CHECKPOINT_EVERY:
            lib.save_df(pd.DataFrame(rows), ckpt_stem)
            uniq_last = list(dict.fromkeys(last_cats[-6:]))
            plog.write(stage, len(rows), tok_in, tok_out,
                       ", ".join(uniq_last), note=f"checkpoint salvo ({len(rows)} linhas)")
            n_since_save = 0

        uniq_last = list(dict.fromkeys(last_cats[-6:]))
        plog.maybe_heartbeat(stage, len(rows), tok_in, tok_out,
                             ", ".join(uniq_last))

    df_results = pd.DataFrame(rows)
    lib.save_df(df_results, ckpt_stem)
    return df_results


# ============================================================
# Orquestração
# ============================================================

def make_run_dir(smoke=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_tag = lib.MODEL_NAME.split("/")[-1]
    suffix = "_smoke" if smoke else ""
    run_dir = Path("results") / f"run_{ts}_{model_tag}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def verify_smoke_permutations(df_expanded):
    """Verifica que as 3 permutações de um logical_id têm contexto idêntico e
    opções rotacionadas (unknown em C, A e B)."""
    for lid, g in df_expanded.groupby("logical_id"):
        assert g["context"].nunique() == 1, f"{lid}: contexto difere entre permutações"
        assert g["question"].nunique() == 1, f"{lid}: pergunta difere entre permutações"
        assert set(g["unknown_position"]) == {"A", "B", "C"}, f"{lid}: unknown não rotaciona por A/B/C"
        # o conjunto de opções (conteúdo) é o mesmo, só muda a posição
        opt_sets = g.apply(lambda r: frozenset([r["option_A"], r["option_B"], r["option_C"]]), axis=1)
        assert opt_sets.nunique() == 1, f"{lid}: conjunto de opções difere entre permutações"
    print(f"OK — {df_expanded['logical_id'].nunique()} exemplos lógicos com permutações coerentes.")


def compute_and_save_metrics(df, run_dir, with_ci=True, n_boot=1000):
    M.metrics_table(df, [], with_ci=with_ci, n_boot=n_boot).to_csv(run_dir / "metrics_overall.csv", index=False)
    M.metrics_table(df, ["category"], with_ci=with_ci, n_boot=n_boot).to_csv(run_dir / "metrics_by_category.csv", index=False)
    M.metrics_table(df, ["category", "bias_target_label"]).to_csv(run_dir / "metrics_by_label.csv", index=False)
    M.unknown_by_position(df).to_csv(run_dir / "metrics_by_position.csv", index=False)


def run(smoke=False, model=None, tokenizer=None, batch_size=BATCH_SIZE):
    torch.manual_seed(lib.RANDOM_SEED)
    np.random.seed(lib.RANDOM_SEED)

    # ---- dados (fonte BR) ----
    df_templates, df_logical, df_expanded = data.build_all()
    if smoke:
        keep = df_logical["logical_id"].head(SMOKE_N_LOGICAL)
        df_logical = df_logical[df_logical["logical_id"].isin(keep)].reset_index(drop=True)
        df_expanded = data.expand_permutations(df_logical)
    verify_smoke_permutations(df_expanded)

    run_dir = make_run_dir(smoke=smoke)
    df_templates.to_csv(run_dir / "base_templates.csv", index=False, encoding="utf-8")
    df_expanded.to_parquet(run_dir / "expanded_examples.parquet", index=False)

    total_forwards = len(df_expanded)
    config = {
        "phase": "smoke" if smoke else "br_full",
        "benchmark": "BR-BBQ (BR-específico)",
        "data_source": data.DATA_JSONL,
        "model_name": lib.MODEL_NAME,
        "dtype": lib.DTYPE,
        "seed": lib.RANDOM_SEED,
        "eval_method": "logprob_first_token",
        "prompt_fix": "unknown por conteudo, sem letra fixa",
        "categories": sorted(df_logical["category"].unique().tolist()),
        "n_logical_examples": int(len(df_logical)),
        "n_expanded_examples": int(total_forwards),
        "n_permutations": 3,
        "batch_size": batch_size,
        "audit_generation": "todos os exemplos",
        "max_new_tokens_audit": lib.MAX_NEW_TOKENS_AUDIT,
        "do_sample": False,
        "checkpoint_every": CHECKPOINT_EVERY,
        "timestamp": datetime.now().isoformat(),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    plog = ProgressLog(run_dir / "PROGRESS.md", total_forwards, config)

    if model is None:
        print("Carregando modelo...")
        tokenizer, model = lib.load_model()

    stage = "smoke test" if smoke else "rodada completa"
    plog.write(stage, 0, 0, 0, "—", note="modelo carregado, iniciando inferência")

    t0 = time.time()
    df_results = run_scoring_br(df_expanded, model, tokenizer, run_dir, plog, batch_size=batch_size, stage=stage)
    elapsed = time.time() - t0

    # ---- pós-processamento (reuso do pipeline) ----
    usage = lib.summarize_tokens(df_results, run_dir)
    agreement, df_audit = lib.audit_agreement(df_results)
    cols = ["logical_id", "category_code", "perm_index", "context_type", "question_type",
            "unknown_position", "correct_option", "predicted_option",
            "parsed_generation_option", "raw_generation",
            "logprob_A", "logprob_B", "logprob_C"]
    df_audit[[c for c in cols if c in df_audit.columns]].to_csv(
        run_dir / "audit_generation_vs_logprob.csv", index=False, encoding="utf-8")
    if agreement:
        (run_dir / "audit_agreement.json").write_text(
            json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8")

    plog.write("métricas", len(df_results), usage["total_input_tokens"],
               usage["total_output_tokens"], "todas", note="scoring concluído, calculando métricas")

    with_ci = not smoke
    compute_and_save_metrics(df_results, run_dir, with_ci=with_ci, n_boot=1000)

    plog.write("relatório", len(df_results), usage["total_input_tokens"],
               usage["total_output_tokens"], "todas", note="gerando REPORT.md e figuras")

    make_report.build_report(run_dir)
    append_br_addendum(run_dir, df_results, usage, agreement, config, elapsed)

    plog.write("relatório", len(df_results), usage["total_input_tokens"],
               usage["total_output_tokens"], "todas",
               note=f"CONCLUÍDO em {elapsed/60:.1f} min — resumo final")

    print(f"\n[{stage}] concluído em {elapsed/60:.1f} min — resultados em {run_dir}")
    print("Tokens:", usage)
    if agreement:
        print("Auditoria geração vs logprob:", agreement)
    print_terminal_summary(df_results, usage, agreement, run_dir)
    return run_dir, df_results


# ============================================================
# Addendum BR no REPORT.md (comparação + baixa potência por categoria)
# ============================================================

def _prev_full_run():
    """Localiza a rodada anterior (seed de 32 templates) em results/, se houver."""
    cand = sorted(Path("results").glob("run_*_full"))
    for d in reversed(cand):
        pq = d / "raw_predictions.parquet"
        if pq.exists():
            return d, pd.read_parquet(pq)
    return None, None


def append_br_addendum(run_dir, df, usage, agreement, config, elapsed):
    run_dir = Path(run_dir)
    by_cat = pd.read_csv(run_dir / "metrics_by_category.csv")
    L = []
    L.append("\n---\n")
    L.append("## Adendo BR-BBQ\n")
    L.append(f"Benchmark **{config['benchmark']}** — 4 categorias BR-específicas: "
             "Religião (REL), Origem regional (REG), Território/Moradia (TER), "
             "Preconceito linguístico (LIN). "
             f"{config['n_logical_examples']} exemplos lógicos × 3 permutações = "
             f"{config['n_expanded_examples']} forwards. Geração livre de auditoria em todos.\n")

    # (4) baixa potência por categoria
    L.append("### Potência estatística por categoria\n")
    L.append("> ⚠️ **Baixa potência**: cada categoria tem apenas n=18 exemplos lógicos "
             "(×3 permutações). Os ICs 95% por categoria são largos; leia as diferenças "
             "entre categorias como exploratórias, não conclusivas.\n")

    code_of = data.CODE_OF_CATEGORY
    show = ["category", "n", "accuracy_ambiguous", "unknown_rate_ambiguous",
            "bias_score_disambiguated", "bias_score_disambiguated_ci_low", "bias_score_disambiguated_ci_high",
            "bias_score_ambiguous", "bias_score_ambiguous_ci_low", "bias_score_ambiguous_ci_high"]
    tbl = by_cat[[c for c in show if c in by_cat.columns]].copy()
    tbl.insert(0, "cód", tbl["category"].map(code_of))

    def _f(x, nd=3):
        return "—" if pd.isna(x) else f"{x:.{nd}f}"

    L.append("| cód | categoria | n | acc_amb | unk_amb | s_DIS [IC95] | s_AMB [IC95] |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in tbl.iterrows():
        sdis = _f(r.get("bias_score_disambiguated"))
        sdis_ci = f" [{_f(r.get('bias_score_disambiguated_ci_low'))}, {_f(r.get('bias_score_disambiguated_ci_high'))}]" if "bias_score_disambiguated_ci_low" in tbl.columns and pd.notna(r.get("bias_score_disambiguated_ci_low")) else ""
        samb = _f(r.get("bias_score_ambiguous"))
        samb_ci = f" [{_f(r.get('bias_score_ambiguous_ci_low'))}, {_f(r.get('bias_score_ambiguous_ci_high'))}]" if "bias_score_ambiguous_ci_low" in tbl.columns and pd.notna(r.get("bias_score_ambiguous_ci_low")) else ""
        L.append(f"| {r['cód']} | {r['category']} | {int(r['n'])} | "
                 f"{_f(r.get('accuracy_ambiguous'))} | {_f(r.get('unknown_rate_ambiguous'))} | "
                 f"{sdis}{sdis_ci} | {samb}{samb_ci} |")
    L.append("")

    # (4) comparação qualitativa com a rodada anterior
    L.append("### Comparação qualitativa com a rodada anterior (seed de 32 templates)\n")
    prev_dir, prev = _prev_full_run()
    if prev is not None:
        def letter_rates(d):
            n = len(d)
            return {l: (d["predicted_option"] == l).mean() for l in ["A", "B", "C"]}
        pr, cr = letter_rates(prev), letter_rates(df)
        prev_amb = prev[prev.context_type == "ambiguous"]
        cur_amb = df[df.context_type == "ambiguous"]
        pu = (prev_amb["predicted_option"] == prev_amb["unknown_position"]).mean() if len(prev_amb) else float("nan")
        cu = (cur_amb["predicted_option"] == cur_amb["unknown_position"]).mean() if len(cur_amb) else float("nan")
        L.append(f"Rodada anterior: `{prev_dir.name}` "
                 f"({prev['category'].nunique()} categoria(s), {len(prev)} forwards; "
                 "note que essa rodada foi interrompida/parcial).\n")
        L.append("| métrica | rodada anterior (seed) | BR-BBQ |")
        L.append("|---|---|---|")
        L.append(f"| taxa letra A / B / C | {pr['A']:.2f} / {pr['B']:.2f} / {pr['C']:.2f} | "
                 f"{cr['A']:.2f} / {cr['B']:.2f} / {cr['C']:.2f} |")
        L.append(f"| unknown rate (ambíguo) | {pu:.3f} | {cu:.3f} |")
        L.append(f"| accuracy geral | {prev['is_correct'].mean():.3f} | {df['is_correct'].mean():.3f} |")
        L.append("")
        L.append("_Comparação apenas qualitativa: as categorias e o número de exemplos "
                 "diferem entre as rodadas; serve para checar se o padrão de viés "
                 "posicional (preferência marginal por uma letra) se mantém._\n")
    else:
        L.append("_Nenhuma rodada anterior com `raw_predictions.parquet` encontrada em `results/`._\n")

    with (run_dir / "REPORT.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(L))


# ============================================================
# Resumo de 10 linhas no terminal
# ============================================================

def print_terminal_summary(df, usage, agreement, run_dir):
    ov = M.all_metrics(df)
    amb = df[df.context_type == "ambiguous"]
    dis = df[df.context_type == "disambiguated"]
    print("\n" + "=" * 60)
    print("RESUMO BR-BBQ (10 linhas)")
    print("=" * 60)
    print(f"1. Forwards: {len(df)} ({df['logical_id'].nunique()} lógicos × 3 perms) | modelo: {lib.MODEL_NAME.split('/')[-1]}")
    print(f"2. Accuracy geral: {ov['accuracy']:.3f} | ambíguo: {ov['accuracy_ambiguous']:.3f} | desambiguado: {ov['accuracy_disambiguated']:.3f}")
    print(f"3. Unknown rate — ambíguo: {ov['unknown_rate_ambiguous']:.3f} (ideal ~1.0) | desambiguado: {ov['unknown_rate_disambiguated']:.3f}")
    print(f"4. Bias score s_DIS: {ov['bias_score_disambiguated']:.3f} | s_AMB: {ov['bias_score_ambiguous']:.3f} (+ = pró-estereótipo)")
    print(f"5. Taxa por letra A/B/C: {ov['letter_rate_A']:.2f} / {ov['letter_rate_B']:.2f} / {ov['letter_rate_C']:.2f} (uniforme=0.33)")
    print(f"6. Position consistency: {ov['position_consistency']:.3f} | flip rate: {ov['flip_rate']:.3f}")
    bycat = df.groupby("category_code")["is_biased_answer"].mean().to_dict()
    print("7. Viés (respostas enviesadas) por categoria: " + " | ".join(f"{k}={v:.2f}" for k, v in sorted(bycat.items())))
    unk_pos = df[df.context_type == "ambiguous"].groupby("unknown_position").apply(
        lambda s: (s["predicted_option"] == s["unknown_position"]).mean())
    print("8. Unknown rate ambíguo por posição: " + " | ".join(f"{k}={v:.2f}" for k, v in unk_pos.items()))
    if agreement:
        print(f"9. Auditoria geração×logprob: concordância {agreement['agreement_on_valid']:.3f} | INVALID {agreement['generation_invalid_rate']:.3f} (n={agreement['n_audited']})")
    else:
        print("9. Auditoria geração×logprob: —")
    print(f"10. Tokens: in {usage['total_input_tokens']:,} / out {usage['total_output_tokens']:,} | {usage['mean_latency_s']:.2f} s/ex | {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    is_smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    run(smoke=is_smoke)
