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
    parser.add_argument("--probes", type=int, nargs="+", help="Explicit IVF nprobe sweep, including matched-recall points")
    args = parser.parse_args()
    if args.nb < 10 or args.nq <= 0 or args.repeats < 3 or not 1 <= args.nlist <= args.nb:
        parser.error("Require nb>=10, nq>0, repeats>=3 and 1<=nlist<=nb")
    if args.probes and any(p <= 0 or p > args.nlist for p in args.probes):
        parser.error("Every probe must be in [1, nlist]")
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
            coll.set_properties({"collection.autocompaction.enabled": "false"})
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
            segment_before = sorted((s.segmentID, s.num_rows) for s in utility.get_query_segment_info(name))
            if sum(rows for _, rows in segment_before) != args.nb:
                raise RuntimeError("Loaded segment row count differs from nb")
            prior_hits = None
            probes = sorted(set((args.probes or [1, 4, 16]) + [args.nlist])) if "IVF" in index else [None]
            probes = [p for p in probes if p is None or p <= args.nlist]
            for probe in probes:
                params = {"metric_type": "L2", "params": {"nprobe": probe} if probe else {}}
                for _ in range(2):
                    coll.search(query, "vec", params, 10, timeout=180)
                samples = []
                windows = []
                sample_results = []
                for _ in range(args.repeats):
                    wall_start_us = time.time_ns() // 1000
                    begin = time.perf_counter()
                    result = coll.search(query, "vec", params, 10, timeout=180)
                    samples.append(time.perf_counter() - begin)
                    windows.append([wall_start_us, time.time_ns() // 1000])
                    # Inspect results outside the timed RPC region, for every sample.
                    sample_ids = [[hit.id for hit in hits] for hits in result]
                    sample_scores = [[float(hit.distance) for hit in hits] for hits in result]
                    if len(sample_ids) != args.nq or any(len(hits) != 10 for hits in sample_ids):
                        raise RuntimeError("Search returned an unexpected nq/top-k shape")
                    if any(len(set(hits)) != 10 for hits in sample_ids):
                        raise RuntimeError("Search returned duplicate IDs")
                    if not np.isfinite(sample_scores).all():
                        raise RuntimeError("Search returned non-finite scores")
                    if sample_results and sample_ids != sample_results[0]["ids"]:
                        raise RuntimeError("Search IDs changed across repeats")
                    sample_results.append({"ids": sample_ids, "scores": sample_scores})
                ids = sample_results[-1]["ids"]
                exact = probe == args.nlist or "IVF" not in index
                if truth is None:
                    truth = ids
                    truth_scores = sample_results[-1]["scores"]
                if exact:
                    truth_by_id = [dict(zip(g, s)) for g, s in zip(truth, truth_scores)]
                    for sample in sample_results:
                        for q, (got_ids, got_scores) in enumerate(zip(sample["ids"], sample["scores"])):
                            if set(got_ids) != set(truth[q]):
                                raise RuntimeError("Exact search IDs differ from FLAT oracle")
                            expected = [truth_by_id[q][id_] for id_ in got_ids]
                            if not np.allclose(got_scores, expected, rtol=1e-4, atol=1e-4):
                                raise RuntimeError("Exact search scores differ from FLAT oracle")
                per_query_hits = [len(set(g) & set(t)) for g, t in zip(truth, ids)]
                recall = sum(per_query_hits) / (args.nq * 10)
                regressions = sum(a < b for a, b in zip(per_query_hits, prior_hits)) if prior_hits else 0
                prior_hits = per_query_hits
                segment_after = sorted((s.segmentID, s.num_rows) for s in utility.get_query_segment_info(name))
                if segment_after != segment_before:
                    raise RuntimeError("Segment layout changed during the timed sweep")
                row = {"index": index, "nprobe": probe, "build_s": build_s,
                       "segments": segment_before, "sample_results": sample_results,
                       "score_tolerance": {"rtol": 1e-4, "atol": 1e-4},
                       "samples_s": samples, "request_windows_us": windows,
                       "batch_p50_ms": float(np.median(samples) * 1000),
                       "qps": args.nq / float(np.median(samples)), "recall_at_10": recall,
                       "ordered_align": sum(a == b for g, t in zip(truth, ids) for a, b in zip(g, t)) / (args.nq * 10),
                       "per_query_hits": per_query_hits, "recall_regression_queries": regressions,
                       "result_ids": ids}
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: value for key, value in row.items() if key not in ("result_ids", "per_query_hits", "sample_results")}), flush=True)
                if regressions or ((probe == args.nlist or "IVF" not in index) and recall != 1):
                    raise RuntimeError("Recall monotonicity/exactness failed")
        finally:
            coll.release()
    print("PASS: RPC nprobe sweep; collections retained for paired revision reload", flush=True)


if __name__ == "__main__":
    main()
