#!/usr/bin/env python3
# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Explicit isolated-service IVF tile validation; never drops collections."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--collection', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--nb', type=int, default=100000)
    ap.add_argument('--repeats', type=int, default=120)
    args = ap.parse_args()
    if args.port != 19540 or not args.collection.startswith('ivftile_') or args.nb < 1024 or args.repeats < 1 or args.output.exists():
        ap.error('isolated port 19540, ivftile_ collection, nb >= 1024, repeats >= 1 and new output required')
    connections.connect(host='127.0.0.1', port=str(args.port))
    rng = np.random.default_rng(42)
    base = rng.uniform(-1, 1, (args.nb, 1024)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    queries = rng.uniform(-1, 1, (32, 1024)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    # Ten interleaved owners give 10%, 50%, 90% and empty eligible sets.
    if args.prepare:
        if utility.has_collection(args.collection):
            raise RuntimeError('Refusing to overwrite existing collection')
        coll = Collection(args.collection, CollectionSchema([
            FieldSchema('id', DataType.INT64, is_primary=True),
            FieldSchema('owner', DataType.VARCHAR, max_length=16),
            FieldSchema('vector', DataType.FLOAT_VECTOR, dim=1024)]))
        coll.set_properties({'collection.autocompaction.enabled': 'false'})
        for start in range(0, args.nb, 2000):
            stop = min(start + 2000, args.nb)
            coll.insert([list(range(start, stop)), ['tenant' + str(i % 10) for i in range(start, stop)],
                         base[start:stop].tolist()])
        coll.flush()
        coll.create_index('vector', {'index_type': 'GPU_IVF_FLAT', 'metric_type': 'IP',
                                    'params': {'nlist': 1024, 'use_device_assign': True}})
    else:
        if not utility.has_collection(args.collection):
            raise RuntimeError('Missing prepared collection')
        coll = Collection(args.collection)
    coll.load()
    def layout():
        return sorted([{'id': s.segmentID, 'partition': s.partitionID, 'rows': s.num_rows}
                       for s in utility.get_query_segment_info(args.collection)], key=lambda s: s['id'])
    before = layout()
    if sum(s['rows'] for s in before) != args.nb:
        raise RuntimeError('Loaded rows differ from input')
    report = {'nb': args.nb, 'dim': 1024, 'k': 20, 'nprobe': 1024, 'queries': 32,
              'timing_boundary': 'HTTP request through response JSON decode; body encoding and result oracle excluded',
              'campaign_wall_boundary': 'includes client oracle and worker scheduling; not pure server throughput',
              'collection': args.collection, 'layout': before, 'index': [i.params for i in coll.indexes],
              'base_sha256': hashlib.sha256(base.tobytes()).hexdigest(),
              'query_sha256': hashlib.sha256(queries.tobytes()).hexdigest(), 'rows': []}
    endpoint = 'http://127.0.0.1:' + str(args.port) + '/v2/vectordb/entities/search'
    def search(body):
        started_us = time.time_ns() // 1000
        started = time.perf_counter()
        req = urllib.request.Request(endpoint, data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
        elapsed = (time.perf_counter() - started) * 1000
        if result.get('code') != 0:
            raise RuntimeError('REST search failed: ' + str(result))
        return result['data'], elapsed, [started_us, time.time_ns() // 1000]
    for owners in ([], [0], list(range(5)), list(range(9)), [-1]):
        expr = '' if not owners else 'owner in ' + json.dumps(['tenant' + str(i) for i in owners])
        eligible = np.arange(args.nb) if not owners else np.flatnonzero(np.isin(np.arange(args.nb) % 10, owners))
        # Exact double oracle computed outside timing; deterministic ID ties.
        truths = []
        for q in queries:
            scores = base[eligible].astype(np.float64) @ q.astype(np.float64)
            order = np.lexsort((eligible, -scores))[:20]
            truths.append(eligible[order].tolist())
        bodies = [json.dumps({'collectionName': args.collection, 'data': [q.tolist()],
                             'annsField': 'vector', 'filter': expr, 'limit': 20,
                             'outputFields': ['owner'], 'consistencyLevel': 'Strong',
                             'searchParams': {'metricType': 'IP', 'params': {'nprobe': 1024}}}).encode()
                  for q in queries]
        def checked(qid):
            hits, elapsed, window = search(bodies[qid])
            ids = [h['id'] for h in hits]
            if ids != truths[qid]:
                raise RuntimeError('Ordered IDs differ from exact filtered oracle')
            for h in hits:
                idx = h['id']
                if h.get('owner') != 'tenant' + str(idx % 10):
                    raise RuntimeError('Foreign owner or incorrect output field')
                score = float(base[idx].astype(np.float64) @ queries[qid].astype(np.float64))
                if abs(h['distance'] - score) > 1e-4 * (1 + abs(score)):
                    raise RuntimeError('Score differs from double oracle')
            return {'qid': qid, 'ms': elapsed, 'window_us': window, 'ids': ids}
        for concurrency in (1, 8):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(pool.map(checked, range(8)))
                started = time.perf_counter()
                samples = list(pool.map(checked, [i % 32 for i in range(args.repeats)]))
                wall = time.perf_counter() - started
            report['rows'].append({'owners': owners, 'eligible': len(eligible), 'concurrency': concurrency,
                                   'wall_s': wall, 'samples': samples, 'truth': truths})
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    if layout() != before:
        raise RuntimeError('Segment layout changed')
    coll.release()
    print('PASS: fixed layout, REST C1/C8, filtered exact IDs/scores; collection retained')


if __name__ == '__main__':
    main()
