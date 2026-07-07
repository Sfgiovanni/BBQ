# results/

Saídas das rodadas de avaliação. Cada rodada fica em
`run_<timestamp>_<modelo>_<fase>/` e contém:

| arquivo | conteúdo |
|---|---|
| `config.json` | configuração da rodada (modelo, seed, nº de exemplos, GPU…) |
| `base_templates.csv` / `expanded_examples.parquet` | dados de entrada gerados |
| `raw_predictions.csv` / `.parquet` | 1 linha por (exemplo lógico × permutação) com logprobs |
| `metrics_*.csv` | métricas agregadas (overall, por categoria, posição, cenário…) |
| `audit_*.{csv,json}` | auditoria geração-livre × logprob |
| `token_usage.json` | contagem de tokens e latência |
| `REPORT.md` + `figures/` | relatório legível e gráficos |

## O que está versionado

Para manter o repositório leve, **só a rodada `full` final** é versionada, em
[`final/`](final/) — e mesmo assim **sem** as predições brutas (`raw_predictions.*`,
`expanded_examples.parquet`), que são pesadas e regeráveis. As demais rodadas
(smoke/pilot/testes antigos) ficam fora do controle de versão (ver `.gitignore`).

Para regenerar qualquer rodada, use os drivers descritos no
[README principal](../README.md).
