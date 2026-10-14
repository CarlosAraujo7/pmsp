"""
pmsp/seeding.py (Entrada 102)
Solucao inicial "melhor das construtivas": a de menor makespan entre LPT, MULTIFIT e LDM
(empate -> a primeira na ordem LPT, MULTIFIT, LDM). Usada pelos metodos *_seed, que partem
dessa solucao em vez de so' do LPT; como o IG nunca piora a solucao corrente, o resultado e'
no maximo o da melhor construtiva por construcao.
"""
from __future__ import annotations

from .constructive import lpt
from .ldm import ldm
from .multifit import multifit


def best_constructive(instance):
    cands = [lpt(instance), multifit(instance), ldm(instance)]
    return min(cands, key=lambda s: s.makespan)
