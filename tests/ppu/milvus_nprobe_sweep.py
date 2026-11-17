#!/usr/bin/env python3
# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Milvus RPC sweep: persistent isolated collections, paired queries, raw samples."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="19540")
    parser.add_argument("--nb", type=int, default=100000)
    parser.add_argument("--nq", type=int, default=100)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    connections.connect(host="127.0.0.1", port=args.port)
    rng = np.random.default_rng(42)
    base = rng.uniform(-1, 1, (args.nb, 128)).astype(np.float32)
    query = rng.uniform(-1, 1, (args.nq, 128)).astype(np.float32).tolist()
    report = {"revision": args.revision, "server": utility.get_server_version(),
              "nb": args.nb, "nq": args.nq, "dim": 128, "k": 10,
              "nlist": args.nlist, "seed": 42, "concurrency": 1,
              "timing_boundary": "pymilvus RPC incl serialization, service, D2H; query list conversion excluded",
              "rows": []}
    truth = None
    for index in ("FLAT", "IVF_FLAT", "FL_BRUTE_FORCE", "FL_IVF_FLAT"):
        name = f"{args.prefix}_{args.nb}_{index.lower()}"
        if args.prepare:
            if utility.has_collection(name):
                raise RuntimeError(f"Refusing to overwrite collection {name}")
            fields = [FieldSchema("id", DataType.INT64, is_primary=True),
                      FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)]
            coll = Collection(name, CollectionSchema(fields))
            for start in range(0, args.nb, 10000):
                stop = min(start + 10000, args.nb)
                coll.insert([list(range(start, stop)), base[start:stop].tolist()])
            coll.flush()
            params = {"nlist": args.nlist} if "IVF" in index else {}
            begin = time.perf_counter()
            coll.create_index("vec", {"index_type": index, "metric_type": "L2", "params": params})
            build_s = time.perf_counter() - begin
        else:
            coll = Collection(name)
            build_s = None
        coll.load()
        try:
            prior_hits = None
            probes = sorted(set((1, 4, 16, args.nlist))) if "IVF" in index else [None]
            for probe in probes:
                params = {"metric_type": "L2", "params": {"nprobe": probe} if probe else {}}
                for _ in range(2):
                    coll.search(query, "vec", params, 10, timeout=180)
                samples = []
                for _ in range(args.repeats):
                    begin = time.perf_counter()
                    result = coll.search(query, "vec", params, 10, timeout=180)
                    samples.append(time.perf_counter() - begin)
                ids = [[hit.id for hit in hits] for hits in result]
                if truth is None:
                    truth = ids
                per_query_hits = [len(set(g) & set(t)) for g, t in zip(truth, ids)]
                recall = sum(per_query_hits) / (args.nq * 10)
                regressions = sum(a < b for a, b in zip(per_query_hits, prior_hits)) if prior_hits else 0
                prior_hits = per_query_hits
                row = {"index": index, "nprobe": probe, "build_s": build_s,
                       "samples_s": samples, "batch_p50_ms": float(np.median(samples) * 1000),
                       "qps": args.nq / float(np.median(samples)), "recall_at_10": recall,
                       "ordered_align": sum(a == b for g, t in zip(truth, ids) for a, b in zip(g, t)) / (args.nq * 10),
                       "per_query_hits": per_query_hits, "recall_regression_queries": regressions,
                       "result_ids": ids}
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: value for key, value in row.items() if key not in ("result_ids", "per_query_hits")}), flush=True)
                if regressions or ((probe == args.nlist or "IVF" not in index) and recall != 1):
                    raise RuntimeError("Recall monotonicity/exactness failed")
        finally:
            coll.release()
    print("PASS: RPC nprobe sweep; collections retained for paired revision reload", flush=True)


if __name__ == "__main__":
    main()
