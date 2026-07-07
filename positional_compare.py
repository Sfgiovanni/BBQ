# -*- coding: utf-8 -*-
"""
Anexa ao REPORT.md de uma rodada nova uma seção comparando as métricas de viés
posicional ANTES (rodada confundida pelo bug do prompt "Escolha C") e DEPOIS
(rodada com o prompt corrigido: unknown referido por conteúdo, sem letra fixa).

Uso:
    python positional_compare.py <run_dir_nova> <run_dir_antiga> [rótulo]
"""

import sys
from pathlib import Path

import pandas as pd

import metrics as M


def positional_summary(df):
    """Extrai as métricas de viés posicional de um DataFrame de predições."""
    pos = M.positional_metrics(df)
    amb = df[df["context_type"] == "ambiguous"]
    unk_amb = float((amb["predicted_option"] == amb["unknown_position"]).mean()) if len(amb) else float("nan")
    by_pos = {}
    for p, s in amb.groupby("unknown_position"):
        by_pos[p] = float((s["predicted_option"] == s["unknown_position"]).mean())
    return {
        "n": len(df),
        "n_logical": int(df["logical_id"].nunique()),
        "letter_rate_A": pos["letter_rate_A"],
        "letter_rate_B": pos["letter_rate_B"],
        "letter_rate_C": pos["letter_rate_C"],
        "position_consistency": pos["position_consistency"],
        "flip_rate": pos["flip_rate"],
        "unknown_rate_ambiguous": unk_amb,
        "unk_amb_pos_A": by_pos.get("A", float("nan")),
        "unk_amb_pos_B": by_pos.get("B", float("nan")),
        "unk_amb_pos_C": by_pos.get("C", float("nan")),
    }


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def build_comparison_md(new_dir, old_dir, old_label):
    new_dir, old_dir = Path(new_dir), Path(old_dir)
    new = positional_summary(pd.read_parquet(new_dir / "raw_predictions.parquet"))
    old = positional_summary(pd.read_parquet(old_dir / "raw_predictions.parquet"))

    L = []
    L.append("\n---\n")
    L.append("## Comparação de viés posicional: antes vs. depois da correção do prompt\n")
    L.append(f"- **Antes** (confundida pelo bug \"Escolha C\"): `{old_dir.name}` "
             f"— {old['n']} forwards, {old['n_logical']} exemplos lógicos. {old_label}")
    L.append(f"- **Depois** (prompt corrigido — unknown por conteúdo, sem letra fixa): `{new_dir.name}` "
             f"— {new['n']} forwards, {new['n_logical']} exemplos lógicos.\n")
    L.append("> O prompt antigo mandava \"Escolha C\" nas três permutações, mesmo quando o "
             "\"Não é possível determinar\" estava em A ou B — induzindo (em vez de medir) "
             "viés posicional para C. A correção referencia o *unknown* pelo conteúdo.\n")

    rows = [
        ("taxa letra A", "letter_rate_A"),
        ("taxa letra B", "letter_rate_B"),
        ("taxa letra C", "letter_rate_C"),
        ("position_consistency", "position_consistency"),
        ("flip_rate", "flip_rate"),
        ("unknown rate (ambíguo, geral)", "unknown_rate_ambiguous"),
        ("unknown rate — unknown em A", "unk_amb_pos_A"),
        ("unknown rate — unknown em B", "unk_amb_pos_B"),
        ("unknown rate — unknown em C", "unk_amb_pos_C"),
    ]
    L.append("| métrica de viés posicional | antes (bug) | depois (corrigido) |")
    L.append("|---|---|---|")
    for label, key in rows:
        L.append(f"| {label} | {_f(old[key])} | {_f(new[key])} |")
    L.append("")

    # leitura automática do efeito
    spread_old = max(old["letter_rate_A"], old["letter_rate_B"], old["letter_rate_C"]) - \
                 min(old["letter_rate_A"], old["letter_rate_B"], old["letter_rate_C"])
    spread_new = max(new["letter_rate_A"], new["letter_rate_B"], new["letter_rate_C"]) - \
                 min(new["letter_rate_A"], new["letter_rate_B"], new["letter_rate_C"])
    L.append(f"- Amplitude da taxa por letra (máx − mín): **{_f(spread_old)}** antes → "
             f"**{_f(spread_new)}** depois (mais próximo de 0 = menos viés posicional).")
    L.append("- Se o unknown rate no ambíguo deixou de depender fortemente da posição "
             "(colunas A/B/C acima mais parecidas entre si), o viés que aparecia antes era "
             "em boa parte artefato do prompt, não do modelo.\n")
    return "\n".join(L)


if __name__ == "__main__":
    new_dir = sys.argv[1]
    old_dir = sys.argv[2]
    old_label = sys.argv[3] if len(sys.argv) > 3 else ""
    md = build_comparison_md(new_dir, old_dir, old_label)
    report = Path(new_dir) / "REPORT.md"
    with report.open("a", encoding="utf-8") as f:
        f.write(md)
    print(f"Seção de comparação anexada a {report}")
