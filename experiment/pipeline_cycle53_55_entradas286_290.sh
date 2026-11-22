#!/usr/bin/env bash
# Ciclos 53 a 55 (Entradas 286 a 290), 22/11/2026: a execucao do plano da segunda revisao cega do artigo do CEC (revisao_cega_cec_2026-11-22.md).
# Rodar da raiz do repositorio. As previsoes e os hashes de cada etapa estao congelados no diario e em results/entrada28[6-9]_prereg_hashes.txt e results/entrada290_prereg_hashes.txt:
# NAO rodar as analises antes de conferir os hashes (md5sum -c) e, na etapa do DE, congelar results/entrada289_twin_predictions.csv ANTES do estagio learn.
set -euo pipefail

# ---- Ciclo 53 (Entradas 286, 286b): estimandos, inferencia e acoplamento em dados existentes; nada novo e' rodado
python -m analysis.report_entrada286_estimands
python -m analysis.report_entrada286b_pool_sensitivity              # POS HOC

# ---- Ciclo 54, parte A (Entradas 287, 287b): registro por decisao do controle aleatorio nas 69 celulas e o modelo de aprendibilidade
python -m pmsp.make_logged_cores                                    # gera pmsp/ig_logged.py e pmsp/ur_logged.py (transformacao exata de texto)
python -m experiment.run_cycle54_logs_entrada287 --all --workers 10 # 34.500 execucoes, ~4 min; results/logs287/*.npz (nao vai para o git)
python -m analysis.report_entrada287_learnability                   # o portao do enquadramento preditivo
python -m analysis.report_entrada287b_posthoc                       # POS HOC

# ---- Ciclo 54, parte B (Entrada 288): teste de invariancia a ordem e contabilidade da linha zerada
python -m experiment.run_cycle54b_order_entrada288 --workers 10     # 27.360 execucoes
python -m analysis.report_entrada288_order

# ---- Ciclo 54, parte C (Entrada 290): dose-resposta de horizonte no R||Cmax B5 (T = 1.000, 4.000, 16.000)
python -m experiment.run_cycle54c_horizon_entrada290 --workers 7
python -m analysis.report_entrada290_horizon

# ---- Ciclo 55 (Entrada 289): evolucao diferencial com selecao de estrategia nas funcoes BBOB (pip install cma==4.6.0)
python -m experiment.run_cycle55_de_entrada289 --stage tune --workers 10
python -m experiment.run_cycle55_de_entrada289 --stage base --workers 10
python -m analysis.predict_entrada289_twin                          # previsoes do gemeo: CONGELAR (hash) antes do proximo estagio
python -m experiment.run_cycle55_de_entrada289 --stage learn --workers 10
python -m analysis.report_entrada289_de

# ---- auditoria numerica dos numeros do diario
python -m analysis.audit_entradas286_290
