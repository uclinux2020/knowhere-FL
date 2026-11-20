"""Run serial/grouped centroid-update comparisons without overlapping workloads."""
import argparse
import json
import os
import pathlib
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--executable", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--repeats", type=int, default=30)
args = parser.parse_args()
output = pathlib.Path(args.output)
output.mkdir(parents=True, exist_ok=False)
results = []
for nb, dim, nlist in [(100000, 128, 128), (1000000, 128, 128), (1000000, 128, 1024)]:
    for distribution in ["uniform", "skewed"]:
        paired = {"serial": [], "grouped": []}
        for run, variant in enumerate(["serial", "grouped", "grouped", "serial"]):
            command = ["taskset", "-c", "0-7", args.executable, str(nb), str(dim), str(nlist),
                       str(args.repeats), variant, distribution]
            env = dict(os.environ, OMP_NUM_THREADS="8", OMP_WAIT_POLICY="PASSIVE",
                       OMP_PROC_BIND="true", OMP_PLACES="cores")
            completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            (output / f"{nb}-{nlist}-{distribution}-{run}-{variant}.log").write_text(completed.stdout)
            samples = next(line for line in completed.stdout.splitlines() if line.startswith("SAMPLES_MS"))
            values = list(map(float, samples.split()[1:]))
            median = statistics.median(values)
            paired[variant].append(median)
            results.append({"nb": nb, "dim": dim, "nlist": nlist, "distribution": distribution,
                            "run": run, "variant": variant, "command": command, "samples_ms": values,
                            "p50_ms": median, "p95_ms": sorted(values)[int(.95 * (len(values) - 1))]})
        baseline = statistics.mean(paired["serial"])
        candidate = statistics.mean(paired["grouped"])
        print(nb, nlist, distribution, baseline, candidate, baseline / candidate, flush=True)
(output / "results.json").write_text(json.dumps(results, indent=2))
