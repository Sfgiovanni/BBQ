# Brazilian BBQ — avaliação de viés social em LLMs de português

Adaptação para o português brasileiro do benchmark **BBQ** (*Bias Benchmark for
Question Answering*, [Parrish et al., 2022](https://aclanthology.org/2022.findings-acl.165/)),
usada para medir **viés social** e **viés posicional** em modelos de linguagem.
Cada exemplo é uma pergunta de múltipla escolha (A/B/C) sobre um par de pessoas
de grupos sociais diferentes, em dois tipos de contexto:

- **Ambíguo** — o contexto não permite decidir; a resposta correta é sempre
  *"Não é possível determinar"*. Escolher uma pessoa revela o viés do modelo.
- **Desambiguado** — o contexto indica a resposta certa; mede-se se o modelo
  ainda erra na direção do estereótipo.

## Como avaliamos

- **Scoring por log-probabilidade do primeiro token** (1 *forward pass*): a
  predição é o `argmax` entre os log-probs das letras A/B/C, em vez de geração
  livre + *parser* de regex. Mais estável e barato.
- **Permutação tripla das alternativas** (rotação cíclica): cada exemplo lógico
  é avaliado 3×, variando a posição do *unknown* e dos grupos, para separar
  **viés social** de **viés posicional** (a tendência de escolher sempre a
  mesma *letra*).
- **Auditoria por geração livre**: em um subconjunto, comparamos a predição por
  log-prob com a resposta gerada em texto, para validar o método.
- **Métricas** (ver [`metrics.py`](metrics.py) e a seção "Explicação das
  métricas" em cada `REPORT.md`): accuracy por contexto, *unknown rate*, os
  *bias scores* `s_AMB`/`s_DIS` de Parrish et al. (2022), taxas por letra,
  *position consistency* / *flip rate*, todos com **IC 95% por bootstrap sobre
  `logical_id`** (respeitando a dependência entre as 3 permutações).

## Dois conjuntos de dados / pipelines

| Pipeline | Dados | Driver | Escala |
|---|---|---|---|
| **BBQ v2 (sintético)** | Templates gerados por código: 2 categorias (**Regionalidade**, **Religião**) × 30 cenários × 100 exemplos | [`run_experiment.py`](run_experiment.py) | até 6.000 exemplos lógicos × 3 = 18.000 avaliações |
| **BR-BBQ (curado)** | [`data/brbbq_br.jsonl`](data/brbbq_br.jsonl): 72 exemplos curados em 4 categorias (Religião, Origem regional, Território/Moradia, Preconceito linguístico) | [`run_brbbq.py`](run_brbbq.py) | 72 × 3 = 216 avaliações |

Ambos reutilizam sem modificação o núcleo de inferência
([`bbq_v2_lib.py`](bbq_v2_lib.py)), as métricas ([`metrics.py`](metrics.py)) e o
relatório ([`make_report.py`](make_report.py)).

## Estrutura do repositório

```
.
├── bbq_v2_lib.py          # núcleo: prompts, scoring por logprob, geração de auditoria, I/O
├── metrics.py             # métricas do BBQ (accuracy, unknown rate, bias scores, bootstrap CI)
├── make_report.py         # gera REPORT.md + figuras a partir de uma rodada
├── brbbq_data.py          # carrega o dataset curado BR-BBQ (data/brbbq_br.jsonl)
├── run_experiment.py      # driver do pipeline sintético (fases: smoke / pilot / full)
├── run_brbbq.py           # driver do pipeline curado BR-BBQ (com checkpoint/retomada)
├── positional_compare.py  # compara viés posicional entre duas rodadas
├── BBQ_v2.ipynb           # notebook de exploração (rodar a partir da raiz do repo)
├── data/                  # dados de entrada (JSONL curado + templates-semente)
└── results/
    ├── README.md          # política de versionamento das rodadas
    └── final/             # rodada full canônica: métricas, REPORT.md, figuras
```

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.8
pip install -r requirements.txt
# torch com GPU (CUDA 12.1):
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

O modelo é baixado do Hugging Face. Se for privado/gated, autentique-se antes:

```bash
export HF_TOKEN=hf_xxx          # o código lê HF_TOKEN do ambiente; nunca faça hardcode
```

## Como reproduzir

```bash
# pipeline sintético — comece pelo smoke (30 exemplos) para validar o ambiente
python run_experiment.py smoke      # rápido, sem IC
python run_experiment.py pilot      # 600 exemplos lógicos, com IC
python run_experiment.py full       # todos os 6.000 exemplos lógicos, com IC

# pipeline curado BR-BBQ
python run_brbbq.py                 # dataset completo (72 exemplos)
python run_brbbq.py smoke           # subconjunto de fumaça

# relatório + figuras de uma rodada (gera REPORT.md e figures/*.png)
python make_report.py results/run_<timestamp>_<modelo>_<fase>/

# comparar viés posicional entre duas rodadas
python positional_compare.py <run_dir_nova> <run_dir_antiga> "rótulo"
```

Cada rodada cria `results/run_<timestamp>_<modelo>_<fase>/` com: `config.json`,
predições brutas (`raw_predictions.*`), tabelas de métricas (`metrics_*.csv`),
auditoria (`audit_*`), `token_usage.json`, `REPORT.md` e `figures/`.

## Modelo avaliado

- **`recogna-nlp/bode-7b-alpaca-pt-br-no-peft`** (Bode 7B, LLaMA-based ajustado
  em Alpaca-pt-br), `float16`, seed 42.
- Rodada de referência executada em GPU **NVIDIA RTX 3080 Ti** (com *offload*
  parcial para CPU).

O modelo é configurável em `MODEL_NAME` no topo de
[`bbq_v2_lib.py`](bbq_v2_lib.py) — o restante do pipeline é agnóstico ao modelo.

## Resultados

Relatório completo da rodada final (`full`, **18.000 avaliações**, seed 42) em
[`results/final/REPORT.md`](results/final/REPORT.md), com figuras em
`results/final/figures/`. Números principais (IC 95% bootstrap sobre `logical_id`):

| métrica | valor |
|---|---|
| accuracy geral | 0,448 |
| accuracy ambíguo | 0,319 [0,316, 0,322] |
| accuracy desambiguado | 0,576 [0,568, 0,584] |
| unknown rate (ambíguo) | 0,319 [0,316, 0,322] |
| bias score s_DIS | 0,016 [−0,008, 0,041] |
| bias score s_AMB | −0,001 [−0,009, 0,006] |
| **taxa letra A / B / C** | **0,054 / 0,841 / 0,105** |
| position consistency | 0,106 [0,101, 0,112] |
| flip rate | 0,893 [0,888, 0,899] |

**Achado principal — viés posicional, não estereótipo.** Os *bias scores*
sociais ficam próximos de zero, mas o modelo escolhe a **letra B em 84% das
respostas**, com *flip rate* de 0,89 (a resposta muda de conteúdo quando as
alternativas são permutadas). O "bug" original de *nunca escolher C* era, na
verdade, um artefato **posicional**: com o *unknown* na posição C o modelo o
escolhe em 0,3% dos casos ambíguos; movido para A/B, sobe para ~48%. A auditoria
por geração livre concorda com o scoring por logprob em 94,5% dos casos válidos.

> Números gerados pelo próprio código do repositório
> (`metrics.py` → `make_report.py`); reproduzíveis com os drivers acima.

## Licença e citação

Código sob licença **MIT** (ver [`LICENSE`](LICENSE)). Ao usar este material,
cite também o benchmark original:

> Parrish, A. et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question
> Answering.* Findings of ACL 2022.

> ⚠️ Os *bias scores* medem tendências estatísticas do modelo em cenários
> construídos; não são um julgamento sobre grupos sociais. Os estereótipos nos
> templates existem **para serem testados e mitigados**, não endossados.
