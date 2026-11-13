"""Baixa instancias EUC_2D da TSPLIB (Reinelt, 1991) com 200 <= n <= 2103 de um espelho publico no GitHub (pdrozdowski/TSPLib.Net, copia da TSPLIB95) para esta pasta, e grava o manifesto (nome, n, tipo de distancia, bytes, md5).
Dados apenas: nada do que e' baixado e' executado; os arquivos sao lidos como texto por pmsp/instance_tsp.py. Uso: python external_benchmarks/tsplib/download_tsplib.py"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/"
NAMES = ("kroA200 kroB200 ts225 tsp225 pr226 gil262 pr264 a280 pr299 lin318 rd400 fl417 pr439 pcb442 d493 u574 rat575 p654 d657 u724 rat783 pr1002 u1060 vm1084 pcb1173 d1291 rl1304 rl1323 nrw1379 fl1400 u1432 fl1577 d1655 vm1748 u1817 rl1889 d2103").split()
man = []
for nm in NAMES:
    dest = HERE / f"{nm}.tsp"
    try:
        data = urllib.request.urlopen(BASE + nm + ".tsp", timeout=60).read()
    except Exception as e:  # noqa: BLE001
        print("FALHOU", nm, e)
        continue
    dest.write_bytes(data)
    txt = data.decode("latin-1")
    n = int(re.search(r"DIMENSION\s*:\s*(\d+)", txt).group(1))
    ew = re.search(r"EDGE_WEIGHT_TYPE\s*:\s*(\w+)", txt).group(1)
    man.append(dict(name=nm, n=n, edge_weight_type=ew, bytes=len(data), md5=hashlib.md5(data).hexdigest()))
    print(nm, n, ew, len(data))
(HERE / "manifest_tsplib.json").write_text(json.dumps(dict(source=BASE, instances=man), indent=1))
print("total:", len(man), "instancias,", sum(m["bytes"] for m in man), "bytes")
