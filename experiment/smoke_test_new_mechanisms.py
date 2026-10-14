import sys
import time

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.apriori_select import apriori_run, apriori_select, dup_ratio
from pmsp.racing import racing_run

print("=== Smoke test: pmsp/apriori_select.py + pmsp/racing.py ===")

inst_lowvar = PMSPInstance.random(n=30, m=4, low=1, high=20, seed=42, name="smoke_lowvar")
dr1 = dup_ratio(inst_lowvar)
sel1 = apriori_select(inst_lowvar)
print(f"inst_lowvar: n=30 m=4 low=1 high=20 dup_ratio={dr1:.4f} apriori_select={sel1}")

inst_hivar = PMSPInstance.random(n=30, m=4, low=1, high=10 ** 8, seed=42, name="smoke_hivar")
dr2 = dup_ratio(inst_hivar)
sel2 = apriori_select(inst_hivar)
print(f"inst_hivar:  n=30 m=4 low=1 high=1e8 dup_ratio={dr2:.4f} apriori_select={sel2}")

assert sel1 != sel2 or (dr1 <= 0.1 and dr2 <= 0.1), "esperado dup_ratio contrastante entre lowvar e hivar"

for label, inst in [("lowvar", inst_lowvar), ("hivar", inst_hivar)]:
    print(f"\n--- apriori_run on {label} ---")
    t0 = time.time()
    res = apriori_run(inst, max_iter=200, seed=1000)
    print(f"  framework_escolhido={res['framework_escolhido']} dup_ratio={res['dup_ratio']:.4f}")
    print(f"  makespan={res['makespan']} gap_pct={res['gap_pct']:.4f} "
          f"cpu_time={res['cpu_time']:.4f}s n_improvements={res['n_improvements']}")
    assert res["makespan"] >= inst.lower_bound, "makespan abaixo do lower bound!"
    assert res["cpu_time"] > 0
    assert res["framework_escolhido"] in ("ils", "ig")
    print(f"  wall={time.time() - t0:.2f}s OK")

    print(f"--- racing_run on {label} ---")
    t0 = time.time()
    res2 = racing_run(inst, max_iter=200, seed=1000)
    print(f"  framework_vencedor={res2['framework_vencedor']} "
          f"probe_ils_makespan={res2['probe_ils_makespan']} "
          f"probe_ig_makespan={res2['probe_ig_makespan']} probe_iters={res2['probe_iters']}")
    print(f"  makespan={res2['makespan']} gap_pct={res2['gap_pct']:.4f} "
          f"cpu_time={res2['cpu_time']:.4f}s n_improvements={res2['n_improvements']}")
    assert res2["makespan"] >= inst.lower_bound, "makespan abaixo do lower bound!"
    assert res2["cpu_time"] > 0
    assert res2["framework_vencedor"] in ("ils", "ig")
    print(f"  wall={time.time() - t0:.2f}s OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
