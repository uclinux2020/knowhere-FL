#!/usr/bin/env python3
# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Correlate opt-in per-node wall intervals with isolated sequential RPC windows."""
import argparse
import json
import re
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--service-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text())
    nodes = []
    runtime_pending = {}
    for line in args.service_log.read_text(errors="replace").splitlines():
        if "FL_IVF_SCAN" not in line and "PPU_PROFILE" not in line:
            continue
        fields = {key: float(value) for key, value in re.findall(r"(\w+)=([0-9]+(?:\.[0-9]+)?)", line)}
        if "PPU_PROFILE" in line and "base_ptr" in fields:
            runtime_pending.setdefault(fields["base_ptr"], []).append(fields)
        elif "start_us" in fields and "elapsed_ms" in fields:
            queue = runtime_pending.get(fields.get("base_ptr"), [])
            # Allocators can reuse BF/assignment base addresses. Require the
            # candidate count as well, and consume the most recent completion.
            matching = [i for i, runtime in enumerate(queue)
                        if runtime.get("candidate_count") == fields.get("candidates")
                        and not (fields.get("full_scan") == 1 and runtime.get("candidate_count") != 0)]
            if matching:
                runtime = queue.pop(matching[-1])
                for key in ("candidate_h2d_ms", "d2h_ms", "workspace_alloc_ms", "launch_wait_ms",
                            "runtime_ms", "score_ms", "select_ms", "batch_calls"):
                    fields[key] = runtime[key]
            nodes.append(fields)
    reports = []
    for row in requests["rows"]:
        if row["index"] != "FL_IVF_FLAT":
            continue
        samples = []
        for start, end in row["request_windows_us"]:
            matches = [node for node in nodes if start <= node["start_us"] <= end
                       and node["nprobe"] == row["nprobe"]]
            if not matches:
                raise RuntimeError("No node records correlated to RPC window")
            first = min(node["start_us"] for node in matches)
            last = max(node["start_us"] + node["elapsed_ms"] * 1000 for node in matches)
            # Sum is work, not latency: segment node intervals may overlap.
            span_ms = (last - first) / 1000
            elapsed_sum_ms = sum(node["elapsed_ms"] for node in matches)
            sample = {"rpc_ms": (end - start) / 1000, "node_calls": len(matches),
                      "node_span_ms": span_ms, "node_work_sum_ms": elapsed_sum_ms,
                      "pre_node_ms": (first - start) / 1000,
                      "post_node_ms": (end - last) / 1000,
                      "start_spread_ms": (max(node["start_us"] for node in matches) - first) / 1000,
                      "max_node_ms": max(node["elapsed_ms"] for node in matches),
                      "overlap_work_ratio": elapsed_sum_ms / span_ms if span_ms else 0,
                      "candidate_bytes": sum(node["candidates"] * 8 for node in matches)}
            for field in ("route_ms", "materialize_ms", "query_stage_ms", "scan_ms", "candidate_h2d_ms",
                          "d2h_ms", "workspace_alloc_ms", "launch_wait_ms", "score_ms", "select_ms"):
                if any(field not in node for node in matches):
                    raise RuntimeError(f"Missing correlated runtime field {field}; log may be interleaved")
                sample[field + "_sum"] = sum(node[field] for node in matches)
                sample[field + "_max"] = max(node[field] for node in matches)
            sample["nodes"] = matches
            samples.append(sample)
        report = {"nb": requests["nb"], "nprobe": row["nprobe"], "samples": samples,
                  "median": {key: statistics.median(sample[key] for sample in samples)
                             for key in samples[0] if key != "nodes"}}
        reports.append(report)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(json.dumps({"nb": report["nb"], "nprobe": report["nprobe"], **report["median"]}))


if __name__ == "__main__":
    main()
