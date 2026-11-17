#!/usr/bin/env python3
# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Compare fixed partition layouts with identical rows and exact RPC truth."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="19540")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nb", type=int, default=100000)
    args = parser.parse_args()
    connections.connect(host="127.0.0.1", port=args.port)
    rng = np.random.default_rng(42)
    base = rng.uniform(-1, 1, (args.nb, 128)).astype(np.float32)
    queries = rng.uniform(-1, 1, (100, 128)).astype(np.float32).tolist()
    # Use the previously prepared exact collection with the same seed and rows.
    truth_collection = Collection(f"flfollowup_20261117_{args.nb}_flat")
    truth_collection.load()
    truth = [[h.id for h in hits] for hits in truth_collection.search(
        queries, "vec", {"metric_type": "L2", "params": {}}, 10)]
    truth_collection.release()
    report = {"nb": args.nb, "nq": 100, "dim": 128, "k": 10,
              "seed": 42, "layout_contract": "distinct partitions cannot merge; auto compaction disabled",
              "rows": []}
    for parts in (1, 4):
        name = f"{args.prefix}_{args.nb}_p{parts}"
        if utility.has_collection(name):
            raise RuntimeError(f"Refusing to overwrite collection {name}")
        coll = Collection(name, CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128)]))
        coll.set_properties({"collection.autocompaction.enabled": "false"})
        for part in range(parts):
            partition = f"layout_{part}"
            coll.create_partition(partition)
            start, end = part * args.nb // parts, (part + 1) * args.nb // parts
            for pos in range(start, end, 10000):
                stop = min(pos + 10000, end)
                coll.insert([list(range(pos, stop)), base[pos:stop].tolist()], partition_name=partition)
        coll.flush()
        coll.create_index("vec", {"index_type": "FL_IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
        coll.load()
        try:
            layout = utility.get_query_segment_info(name)
            layout_info = [{"segment_id": s.segmentID, "partition_id": s.partitionID,
                            "rows": s.num_rows} for s in layout]
            for probe in (1, 4, 16, 128):
                params = {"metric_type": "L2", "params": {"nprobe": probe}}
                for _ in range(2):
                    coll.search(queries, "vec", params, 10)
                samples, windows = [], []
                for _ in range(7):
                    start = time.time_ns() // 1000
                    begin = time.perf_counter()
                    result = coll.search(queries, "vec", params, 10)
                    samples.append(time.perf_counter() - begin)
                    windows.append([start, time.time_ns() // 1000])
                ids = [[h.id for h in hits] for hits in result]
                recall = sum(len(set(g) & set(t)) for g, t in zip(truth, ids)) / 1000
                row = {"index": "FL_IVF_FLAT", "parts": parts, "nprobe": probe,
                       "segment_layout": layout_info, "request_windows_us": windows,
                       "samples_s": samples, "batch_p50_ms": float(np.median(samples) * 1000),
                       "recall_at_10": recall, "result_ids": ids}
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k not in ("result_ids", "request_windows_us")}), flush=True)
                if probe == 128 and recall != 1:
                    raise RuntimeError("Full-probe result is not exact")
            after = utility.get_query_segment_info(name)
            if sorted((s.segmentID, s.num_rows) for s in after) != sorted((s.segmentID, s.num_rows) for s in layout):
                raise RuntimeError("Segment layout changed during measurement")
        finally:
            coll.release()
    print("PASS: fixed layouts, exact full probes; collections retained", flush=True)


if __name__ == "__main__":
    main()
