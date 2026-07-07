# -*- coding: utf-8 -*-
"""Gera REPORT.md + figuras (matplotlib) para um run do BBQ v2."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# paleta categórica validada (dataviz skill): azul, aqua, amarelo
C_BLUE, C_AQUA, C_YELLOW = "#2a78d6", "#1baf7a", "#eda100"
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color="#eceae6", linewidth=0.8)
    ax.set_axisbelow(True)


def _bar(ax, labels, values, colors, fmt="{:.2f}"):
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, color=INK)


def fig_unknown_by_position(by_pos, figdir):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    d = by_pos.sort_values("unknown_position")
    _bar(ax, "unknown em " + d["unknown_position"], d["unknown_rate_ambiguous"], C_BLUE)
    ax.set_ylim(0, 1.05)
    _style(ax, "Unknown rate no contexto ambíguo, por posição do unknown")
    ax.set_ylabel("taxa de escolha do unknown", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(figdir / "unknown_by_position.png"); plt.close(fig)


def fig_letter_rates(overall, figdir):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    vals = [overall["letter_rate_A"], overall["letter_rate_B"], overall["letter_rate_C"]]
    _bar(ax, ["A", "B", "C"], vals, [C_BLUE, C_AQUA, C_YELLOW])
    ax.axhline(1 / 3, color=INK2, linestyle="--", linewidth=1)
    ax.annotate("uniforme (1/3)", (2.3, 1 / 3), fontsize=8, color=INK2, va="bottom", ha="right")
    ax.set_ylim(0, 1.05)
    _style(ax, "Taxa marginal de escolha de cada letra (todas as permutações)")
    fig.tight_layout(); fig.savefig(figdir / "letter_rates.png"); plt.close(fig)


def fig_bias_by_category(by_cat, figdir):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    x = range(len(by_cat))
    for i, (_, r) in enumerate(by_cat.iterrows()):
        for off, key, color, lab in [(-0.12, "bias_score_disambiguated", C_BLUE, "s_DIS"),
                                     (0.12, "bias_score_ambiguous", C_AQUA, "s_AMB")]:
            lo, hi = r.get(f"{key}_ci_low"), r.get(f"{key}_ci_high")
            v = r[key]
            if pd.notna(lo):
                ax.plot([i + off, i + off], [lo, hi], color=color, linewidth=2)
            ax.plot(i + off, v, "o", color=color, markersize=8,
                    label=lab if i == 0 else None)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(list(x)); ax.set_xticklabels(by_cat["category"])
    ax.set_ylim(-1, 1)
    _style(ax, "Bias score BBQ por categoria (IC 95% bootstrap)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figdir / "bias_by_category.png"); plt.close(fig)


def fig_accuracy(overall, figdir):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    _bar(ax, ["ambíguo", "desambiguado"],
         [overall["accuracy_ambiguous"], overall["accuracy_disambiguated"]],
         [C_BLUE, C_AQUA])
    ax.set_ylim(0, 1.05)
    _style(ax, "Accuracy por tipo de contexto")
    fig.tight_layout(); fig.savefig(figdir / "accuracy_by_context.png"); plt.close(fig)


def _f(x, nd=3):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def _ci(row, key):
    lo, hi = row.get(f"{key}_ci_low"), row.get(f"{key}_ci_high")
    if lo is None or pd.isna(lo):
        return _f(row.get(key))
    return f"{_f(row[key])} [{_f(lo)}, {_f(hi)}]"


def build_report(run_dir):
    run_dir = Path(run_dir)
    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)

    overall = pd.read_csv(run_dir / "metrics_overall.csv").iloc[0]
    by_cat = pd.read_csv(run_dir / "metrics_by_category.csv")
    by_pos = pd.read_csv(run_dir / "metrics_by_position.csv")
    config = json.loads((run_dir / "config.json").read_text())
    usage = json.loads((run_dir / "token_usage.json").read_text())
    agreement = None
    if (run_dir / "audit_agreement.json").exists():
        agreement = json.loads((run_dir / "audit_agreement.json").read_text())

    fig_unknown_by_position(by_pos, figdir)
    fig_letter_rates(overall, figdir)
    fig_bias_by_category(by_cat, figdir)
    fig_accuracy(overall, figdir)

    amb_c = by_pos[by_pos.unknown_position == "C"]["unknown_rate_ambiguous"]
    amb_ab = by_pos[by_pos.unknown_position != "C"]["unknown_rate_ambiguous"]

    lines = []
    lines.append(f"# Relatório — Brazilian BBQ v2 ({config['phase']})\n")
    lines.append(f"Modelo: `{config['model_name']}` | método: scoring por logprobs do 1º token | "
                 f"{config['n_logical_examples']} exemplos lógicos × 3 permutações = {config['n_expanded_examples']} avaliações | seed {config['seed']}.\n")

    lines.append("## Explicação das métricas\n")
    lines.append("- **Accuracy (ambíguo/desambiguado)**: fração de acertos; no contexto ambíguo a resposta correta é sempre o *unknown* (\"Não é possível determinar\").")
    lines.append("- **Unknown rate (ambíguo)**: com que frequência o modelo escolhe o unknown quando deveria; num modelo calibrado seria ~100%.")
    lines.append("- **Bias score s_DIS** (Parrish et al. 2022): nos desambiguados, 2·(respostas enviesadas / respostas não-unknown) − 1; +1 = sempre pró-estereótipo, −1 = anti, 0 = neutro.")
    lines.append("- **Bias score s_AMB**: direção do viés nos ambíguos ponderada pela taxa de erro, (1 − acc_amb)·s_direção.")
    lines.append("- **Taxa por letra**: quanto cada letra A/B/C é escolhida, marginalizada sobre as 3 permutações — deveria ser ~uniforme se o modelo fosse insensível à posição.")
    lines.append("- **Position consistency / flip rate**: fração de exemplos lógicos em que a predição aponta para o mesmo conteúdo nas 3 permutações / em que muda.")
    lines.append("- **IC 95%**: bootstrap sobre `logical_id` (respeita a dependência entre permutações).\n")

    lines.append("## Números principais\n")
    lines.append("| métrica | valor |")
    lines.append("|---|---|")
    lines.append(f"| accuracy geral | {_f(overall['accuracy'])} |")
    lines.append(f"| accuracy ambíguo | {_ci(overall, 'accuracy_ambiguous')} |")
    lines.append(f"| accuracy desambiguado | {_ci(overall, 'accuracy_disambiguated')} |")
    lines.append(f"| unknown rate (ambíguo) | {_ci(overall, 'unknown_rate_ambiguous')} |")
    lines.append(f"| unknown rate (desambiguado) | {_f(overall['unknown_rate_disambiguated'])} |")
    lines.append(f"| bias score s_DIS | {_ci(overall, 'bias_score_disambiguated')} |")
    lines.append(f"| bias score s_AMB | {_ci(overall, 'bias_score_ambiguous')} |")
    lines.append(f"| taxa letra A / B / C | {_f(overall['letter_rate_A'])} / {_f(overall['letter_rate_B'])} / {_f(overall['letter_rate_C'])} |")
    lines.append(f"| position consistency | {_ci(overall, 'position_consistency')} |")
    lines.append(f"| flip rate | {_ci(overall, 'flip_rate')} |\n")

    lines.append("## 1. Viés posicional\n")
    lines.append("![letras](figures/letter_rates.png)\n")
    lines.append("![unknown por posição](figures/unknown_by_position.png)\n")
    lines.append(by_pos.to_markdown(index=False) + "\n")

    lines.append("## 2. Bias scores por categoria (IC 95%)\n")
    lines.append("![bias](figures/bias_by_category.png)\n")
    cols = ["category", "n", "accuracy_ambiguous", "accuracy_disambiguated",
            "bias_score_disambiguated", "bias_score_ambiguous", "unknown_rate_ambiguous"]
    lines.append(by_cat[[c for c in cols if c in by_cat.columns]].to_markdown(index=False) + "\n")
    lines.append("![accuracy](figures/accuracy_by_context.png)\n")

    lines.append("## 3. Diagnóstico do problema original (\"nunca escolhe C\")\n")
    if agreement:
        lines.append(f"- Auditoria geração livre × logprob: {agreement['n_audited']} exemplos auditados, "
                     f"taxa de INVALID na geração = {_f(agreement['generation_invalid_rate'])}, "
                     f"concordância nos válidos = {_f(agreement['agreement_on_valid'])}.")
    if len(amb_c) and len(amb_ab):
        c_rate, ab_rate = float(amb_c.iloc[0]), float(amb_ab.mean())
        lines.append(f"- Unknown rate no ambíguo com unknown fixo em C (layout original): {_f(c_rate)}; "
                     f"com unknown em A ou B: {_f(ab_rate)}.")
        if ab_rate - c_rate > 0.1:
            concl = ("O problema original era em grande parte **posicional**: quando o unknown sai da posição C, "
                     "o modelo passa a escolhê-lo com mais frequência.")
        elif max(c_rate, ab_rate) < 0.15:
            concl = ("O modelo raramente escolhe o unknown em **qualquer** posição — o problema não é a posição C "
                     "nem o parser, mas o próprio modelo, que quase nunca responde \"não é possível determinar\".")
        else:
            concl = "O efeito posicional existe mas é parcial; ver tabela por posição."
        lines.append(f"- **Conclusão**: {concl}\n")

    lines.append("## Custo\n")
    lines.append(f"- Tokens de entrada: {usage['total_input_tokens']:,} | saída: {usage['total_output_tokens']:,} | "
                 f"total: {usage['total_tokens']:,} | média/exemplo: {usage['mean_tokens_per_example']:.0f}.")
    lines.append(f"- Latência média: {usage['mean_latency_s']:.2f} s/exemplo.\n")

    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir / "REPORT.md"


if __name__ == "__main__":
    import sys
    print(build_report(sys.argv[1]))
