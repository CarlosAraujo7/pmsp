"""Figura do manuscrito de periodico: o valor do conhecimento offline por granularidade contra o aprendizado online (Entrada 266, B30).
Uso: python analysis/make_fig_journal_granularity.py     Saida: journal/fig_granularity.pdf (+ png de conferencia na pasta de rascunho se pedido)
Cores (paleta de referencia, tres primeiras posicoes categoricas, validadas em todos os pares): azul = acao estatica escolhida offline; laranja = aprendiz online; verde-agua = politica treinada offline; cinza = aleatorio.
Barras tracejadas = regimes de transferencia (a acao e' escolhida sem os dados do tipo ou do tamanho testado). Cada barra leva o valor como rotulo direto (o verde-agua tem contraste baixo no fundo claro)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
C = pd.read_csv(ROOT / "results" / "entrada266_granularity_cells.csv")
B = C[C.space == "B30"]
m = lambda col: float(B[col].mean())
best = float(B[[f"g_{k}" for k in ("ql", "ucb", "pg", "alns", "qo", "iq", "fr")]].max(axis=1).mean())
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c87", "#0b0b0b", "#52514e"
rows = [  # (rotulo, valor, cor, tracejada)
    ("random choice", 0.0, GREY, False),
    ("standard Q-Learning", m("g_ql"), ORANGE, False),
    ("order-invariant Q-Learning", m("g_ql_perm"), ORANGE, False),
    ("best online learner of each type", best, ORANGE, False),
    ("static: one action for all instances", m("g_G0"), BLUE, False),
    ("static: one action per type", m("g_Gt"), BLUE, False),
    ("static: one action per size", m("g_Gn"), BLUE, False),
    ("static: one action per class (validated)", m("g_Gtn"), BLUE, False),
    ("one action for all, chosen without the type", m("g_LOTO_G0"), BLUE, True),
    ("one action per size, chosen without the type", m("g_LOTO_Gn"), BLUE, True),
    ("one action for all, chosen without the size", m("g_LOSO_G0"), BLUE, True),
    ("one action per type, chosen without the size", m("g_LOSO_Gt"), BLUE, True),
    ("constant mixture trained offline", m("g_nn_const"), AQUA, False),
    ("neural policy trained offline", m("g_nn"), AQUA, False),
]
fig, ax = plt.subplots(figsize=(6.4, 4.3))
y = list(range(len(rows)))[::-1]
for yi, (lab, v, col, hatch) in zip(y, rows):
    ax.barh(yi, v, height=0.62, color=col, edgecolor="white" if not hatch else col, linewidth=0.8, hatch="////" if hatch else None, alpha=0.55 if hatch else 1.0)
    ax.text(v + (0.04 if v >= 0 else -0.04), yi, f"{v:.2f}".replace("-", "−"), va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.5, color=INK)
ax.set_xlabel("gain over random choice, points of makespan (mean of the seven instance types, B30)", fontsize=7.5, color=MUTED)
ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=2)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.95, 2.75)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c2")
ax.xaxis.grid(True, color="#e6e5e0", lw=0.6)
ax.set_axisbelow(True)
from matplotlib.patches import Patch  # noqa: E402

ax.legend(handles=[Patch(color=ORANGE, label="online learner"), Patch(color=BLUE, label="static, chosen offline"), Patch(facecolor=BLUE, alpha=0.55, hatch="////", edgecolor=BLUE, label="static, to an unseen type/size"),
                   Patch(color=AQUA, label="trained offline")], fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
out = ROOT / "journal" / "fig_granularity.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(ROOT / "journal" / "fig_granularity_preview.png", dpi=110, bbox_inches="tight")
print(out)
