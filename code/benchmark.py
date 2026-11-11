"""Compares routed (cost-aware) vs. baseline (always-strong) execution
across the same set of test queries."""

from backend import run_research_query
from baseline import run_baseline_query
from querys import TEST_QUERIES

routed_total_cost = 0.0
routed_total_latency = 0.0
baseline_total_cost = 0.0
baseline_total_latency = 0.0

print(f"Running {len(TEST_QUERIES)} queries through both pipelines...\n")

for i, query in enumerate(TEST_QUERIES, 1):
    print(f"[{i}/{len(TEST_QUERIES)}] {query[:70]}...")

    routed = run_research_query(query)
    baseline = run_baseline_query(query)

    tiers = [r["tier"] for r in routed["results"]]
    cheap_count = tiers.count("cheap")
    strong_count = tiers.count("strong")

    print(f"   Routed:   ${routed['total_cost_usd']:.6f} | {routed['total_latency_sec']:.2f}s | "
          f"{cheap_count} cheap / {strong_count} strong")
    print(f"   Baseline: ${baseline['total_cost_usd']:.6f} | {baseline['total_latency_sec']:.2f}s | "
          f"{len(tiers)} strong (forced)\n")

    routed_total_cost += routed["total_cost_usd"]
    routed_total_latency += routed["total_latency_sec"]
    baseline_total_cost += baseline["total_cost_usd"]
    baseline_total_latency += baseline["total_latency_sec"]

cost_savings_pct = (1 - routed_total_cost / baseline_total_cost) * 100 if baseline_total_cost else 0
latency_savings_pct = (1 - routed_total_latency / baseline_total_latency) * 100 if baseline_total_latency else 0

print("=" * 55)
print("BENCHMARK SUMMARY")
print("=" * 55)
print(f"Routed total cost:    ${routed_total_cost:.6f}")
print(f"Baseline total cost:  ${baseline_total_cost:.6f}")
print(f"Cost savings:         {cost_savings_pct:.1f}%")
print()
print(f"Routed total latency:    {routed_total_latency:.2f}s")
print(f"Baseline total latency:  {baseline_total_latency:.2f}s")
print(f"Latency savings:         {latency_savings_pct:.1f}%")
print("=" * 55)