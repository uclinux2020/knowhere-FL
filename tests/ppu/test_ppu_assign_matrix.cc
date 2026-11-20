// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

namespace ppu = knowhere::fl::ppu;
int
main(int argc, char** argv) {
    const int device = argc > 1 ? std::stoi(argv[1]) : 0;
    if (ppu::SetDevice(device) != 0)
        throw std::runtime_error("Cannot bind requested physical device");
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    constexpr int64_t dim = 128;
    for (int64_t nlist : {31, 128, 1024}) {
        for (int64_t nq : {1, 17, 257}) {
            for (int mode : {0, 1, 2}) {
                std::vector<float> c(nlist * dim), x(nq * dim);
                for (auto& v : c) v = mode == 1 ? 1.f : value(rng);
                for (auto& v : x) v = value(rng);
                if (mode == 2) {
                    for (auto& v : c) v = 10000.f + v;
                    for (auto& v : x) v = 10000.f + v;
                }
                void* dc = ppu::Alloc(c.size() * sizeof(float));
                void* dx = ppu::Alloc(x.size() * sizeof(float));
                if (!dc || !dx)
                    throw std::runtime_error("Allocation failed");
                if (ppu::CopyH2D(dc, c.data(), c.size() * sizeof(float), nullptr) ||
                    ppu::CopyH2D(dx, x.data(), x.size() * sizeof(float), nullptr))
                    throw std::runtime_error("Upload failed");
                std::vector<int64_t> ids(nq);
                if (ppu::Assign(dx, nq, dim, dc, nlist, 0, ids.data(), nullptr))
                    throw std::runtime_error("Native assignment failed");
                for (int64_t q = 0; q < nq; ++q) {
                    double best = 1e300;
                    int64_t expected = -1;
                    for (int64_t j = 0; j < nlist; ++j) {
                        double score = 0;
                        for (int64_t d = 0; d < dim; ++d) {
                            double a = static_cast<double>(x[q * dim + d]) - c[j * dim + d];
                            score += a * a;
                        }
                        if (score < best) {
                            best = score;
                            expected = j;
                        }
                    }
                    if (ids[q] != expected)
                        throw std::runtime_error("Nearest centroid ID differs from double direct oracle");
                }
                ppu::Free(dc);
                ppu::Free(dx);
            }
        }
    }
    std::cout << "PASS: L2 assignment double oracle, random/ties/large-offset, tail query tiles and centroid blocks\n";
}
