# -*- coding: utf-8 -*-
"""
Fonte de dados BR-BBQ (BR-específico).

Substitui a geração de templates/exemplos do bbq_v2_lib pelos arquivos
já expandidos deste diretório:

  - brbbq_br.jsonl        : 72 exemplos no formato BBQ;
  - templates_seed_br.json: 12 templates de referência (proveniência).

NÃO altera o texto dos contextos/perguntas/alternativas: os campos de opção
(inclusive o texto "unknown") vêm diretamente do JSONL. Só a orquestração e o
mapeamento para as colunas do pipeline são feitos aqui, para reusar sem
modificação o scoring (bbq_v2_lib), as métricas (metrics.py) e o relatório
(make_report.py).

O output `df_expanded` tem exatamente as mesmas colunas que
bbq_v2_lib.expand_permutations produz, de modo que o restante do pipeline
funciona sem qualquer refatoração.
"""

import json
from pathlib import Path

import pandas as pd

from bbq_v2_lib import LETTERS, PERM_LAYOUTS, stable_hash

# ============================================================
# Constantes BR
# ============================================================

# Os dados ficam em data/ ao lado deste módulo; caminho resolvido de forma
# independente do diretório de trabalho.
_DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_JSONL = str(_DATA_DIR / "brbbq_br.jsonl")
TEMPLATES_JSON = str(_DATA_DIR / "templates_seed_br.json")

N_EXAMPLES = 72
N_CATEGORIES = 4
EXAMPLES_PER_CATEGORY = 18

# prefixo do item_id -> (nome da categoria no JSONL, código curto)
CATEGORY_CODES = {
    "REL": "Religiao",
    "REG": "Origem regional",
    "TER": "Territorio/Moradia",
    "LIN": "Preconceito linguistico",
}
CODE_OF_CATEGORY = {v: k for k, v in CATEGORY_CODES.items()}

# BBQ -> nomenclatura interna do pipeline
CONTEXT_MAP = {"ambig": "ambiguous", "disambig": "disambiguated"}
POLARITY_MAP = {"neg": "negative", "nonneg": "non_negative"}
# label (índice da resposta correta) -> conteúdo semântico
#   ans0 = target (grupo1), ans1 = nontarget (grupo2), ans2 = unknown
LABEL_TO_CONTENT = {0: "grupo1", 1: "grupo2", 2: "unknown"}
# papel enviesado -> conteúdo (o enviesado nunca é "unknown")
ROLE_TO_CONTENT = {"target": "grupo1", "nontarget": "grupo2"}


# ============================================================
# Carga + asserts de sanidade
# ============================================================

def load_raw(path=DATA_JSONL):
    """Lê o JSONL bruto (uma linha por exemplo)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    _assert_dataset(rows)
    return rows


def _assert_dataset(rows):
    """Verificações de carga exigidas: 72 exemplos; 4 categorias × 18;
    context_condition válido; label coerente (unknown nos ambíguos)."""
    assert len(rows) == N_EXAMPLES, f"esperado {N_EXAMPLES} exemplos, obtido {len(rows)}"

    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    assert len(cats) == N_CATEGORIES, f"esperado {N_CATEGORIES} categorias, obtido {sorted(cats)}"
    for cat, n in cats.items():
        assert n == EXAMPLES_PER_CATEGORY, f"categoria {cat} tem {n} exemplos (esperado {EXAMPLES_PER_CATEGORY})"

    seen_ids = set()
    for r in rows:
        assert r["context_condition"] in CONTEXT_MAP, f"context_condition inválido: {r['context_condition']}"
        assert r["question_polarity"] in POLARITY_MAP, f"question_polarity inválido: {r['question_polarity']}"
        assert r["label"] in (0, 1, 2), f"label inválido: {r['label']}"
        # coerência: em ambíguo o correto é sempre o unknown (ans2, label=2)
        if r["context_condition"] == "ambig":
            assert r["label"] == 2, f"exemplo ambíguo {r['example_id']} deveria ter label=2 (unknown)"
        # papéis das respostas
        assert r["answer_info"]["ans0"][1] == "target"
        assert r["answer_info"]["ans1"][1] == "nontarget"
        assert r["answer_info"]["ans2"][1] == "unknown"
        assert r["example_id"] not in seen_ids, f"example_id duplicado: {r['example_id']}"
        seen_ids.add(r["example_id"])
    return True


def load_reference_templates(path=TEMPLATES_JSON):
    """Carrega os 12 templates de referência como base_templates.csv."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    templates = obj["templates"]
    assert len(templates) == 12, f"esperado 12 templates, obtido {len(templates)}"
    df = pd.DataFrame(templates)
    df["category_code"] = df["id"].str.split("-").str[0]
    return df


# ============================================================
# JSONL -> exemplos lógicos (formato do pipeline)
# ============================================================

def build_logical_examples(rows):
    """Converte cada exemplo do JSONL em uma linha 'lógica' com as colunas que
    o expander/scoring/métricas esperam. Usa os textos reais das alternativas.

    Convenção (idêntica ao BBQ): grupo1 = alvo (ans0), grupo2 = não-alvo (ans1),
    unknown = ans2. O texto de "unknown" é preservado por exemplo.
    """
    examples = []
    for r in rows:
        code = r["item_id"].split("-")[0]
        assert code in CATEGORY_CODES, f"código de categoria desconhecido: {code}"

        grupo1 = r["ans0"]          # alvo (target)
        grupo2 = r["ans1"]          # não-alvo (nontarget)
        unknown_text = r["ans2"]    # texto real do "unknown" (preservado)

        context_type = CONTEXT_MAP[r["context_condition"]]
        question_type = POLARITY_MAP[r["question_polarity"]]
        correct_content = LABEL_TO_CONTENT[r["label"]]
        biased_role = r["additional_metadata"]["biased_answer_role"]
        biased_content = ROLE_TO_CONTENT[biased_role]

        # coerência: em ambíguo o correto é o unknown
        if context_type == "ambiguous":
            assert correct_content == "unknown"

        examples.append({
            "logical_id": f"brbbq_{r['example_id']:03d}",
            "orig_example_id": r["example_id"],
            "template_id": r["item_id"],
            "scenario_name": r["item_id"],
            "category": r["category"],
            "category_code": code,
            "grupo1": grupo1,
            "grupo2": grupo2,
            "unknown_text": unknown_text,
            "bias_target_label": grupo1,          # alvo do estereótipo = ans0
            "non_bias_label": grupo2,
            "bias_target_content": "grupo1",
            "context_type": context_type,
            "question_type": question_type,
            "context": r["context"],
            "question": r["question"],
            "correct_content": correct_content,
            "biased_content": biased_content,
            "stereotype": r["additional_metadata"].get("stereotype", ""),
            "provenance": r["additional_metadata"].get("provenance", ""),
        })

    df = pd.DataFrame(examples)
    assert len(df) == N_EXAMPLES
    assert df["logical_id"].nunique() == len(df)
    return df


# ============================================================
# Permutação tripla (rotação cíclica) — preserva o texto real do unknown
# ============================================================

def expand_permutations(df_logical):
    """3 variantes por exemplo lógico, rotacionando o conteúdo pelas posições
    A/B/C (mesmas PERM_LAYOUTS do pipeline). Recalcula
    correct_option/biased_option/unknown_position por variante.

    Diferença única em relação a bbq_v2_lib.expand_permutations: o texto de
    "unknown" vem da coluna `unknown_text` (do JSONL), em vez de uma constante
    global — para não alterar o conteúdo dos exemplos BR.
    """
    rows = []
    for rec in df_logical.to_dict("records"):
        content_label = {
            "grupo1": rec["grupo1"],
            "grupo2": rec["grupo2"],
            "unknown": rec["unknown_text"],
        }
        for perm_index, layout in PERM_LAYOUTS.items():
            letter_of = {content: LETTERS[i] for i, content in enumerate(layout)}
            row = dict(rec)
            row.update({
                "example_id": stable_hash(f"{rec['logical_id']}|perm{perm_index}"),
                "perm_index": perm_index,
                "option_A": content_label[layout[0]],
                "option_B": content_label[layout[1]],
                "option_C": content_label[layout[2]],
                "unknown_position": letter_of["unknown"],
                "bias_target_position": letter_of[rec["bias_target_content"]],
                "correct_option": letter_of[rec["correct_content"]],
                "biased_option": letter_of[rec["biased_content"]],
                "content_of_A": layout[0],
                "content_of_B": layout[1],
                "content_of_C": layout[2],
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    assert len(df) == 3 * len(df_logical), f"esperado {3*len(df_logical)}, obtido {len(df)}"

    # sanidade: unknown_position coerente com correct_option nos ambíguos
    amb = df[df["context_type"] == "ambiguous"]
    assert (amb["correct_option"] == amb["unknown_position"]).all()
    # sanidade: 3 permutações por logical_id, posições do unknown distintas
    g = df.groupby("logical_id")
    assert (g["perm_index"].nunique() == 3).all()
    assert (g["unknown_position"].nunique() == 3).all()
    # contexto/pergunta idênticos entre permutações do mesmo logical_id
    assert (g["context"].nunique() == 1).all() and (g["question"].nunique() == 1).all()
    # sanidade extra: correct_option coerente com a posição do conteúdo correto
    for rec in df.to_dict("records"):
        assert rec[f"content_of_{rec['correct_option']}"] == rec["correct_content"]
        assert rec[f"content_of_{rec['biased_option']}"] == rec["biased_content"]
    return df


def build_all(jsonl_path=DATA_JSONL, templates_path=TEMPLATES_JSON):
    """Pipeline de dados completo: retorna (df_templates, df_logical, df_expanded)."""
    rows = load_raw(jsonl_path)
    df_templates = load_reference_templates(templates_path)
    df_logical = build_logical_examples(rows)
    df_expanded = expand_permutations(df_logical)
    return df_templates, df_logical, df_expanded


if __name__ == "__main__":
    dt, dl, de = build_all()
    print(f"templates: {len(dt)} | lógicos: {len(dl)} | expandidos: {len(de)}")
    print("categorias:", dl["category"].value_counts().to_dict())
    print("contextos:", de["context_type"].value_counts().to_dict())
    print("OK — asserts de sanidade passaram.")
