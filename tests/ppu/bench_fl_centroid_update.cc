// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/fl/fl_centroid_update.h"

int
main(int argc, char** argv) {
    if (argc != 7)
        throw std::runtime_error("Args: nb dim nlist repeats grouped|serial uniform|skewed");
    const int64_t nb = std::stoll(argv[1]), dim = std::stoll(argv[2]), nlist = std::stoll(argv[3]);
    const int repeats = std::stoi(argv[4]);
    const bool grouped = std::string(argv[5]) == "grouped", skewed = std::string(argv[6]) == "skewed";
    if (nb <= 0 || dim <= 0 || nlist <= 0 || repeats <= 0)
        throw std::runtime_error("Invalid shape");
    std::mt19937 random(42);
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    std::vector<float> x(nb * dim);
    std::vector<int64_t> owner(nb);
    for (auto& v : x) v = value(random);
    for (int64_t j = 0; j < nb; ++j) owner[j] = skewed && j % 10 != 0 ? 0 : random() % nlist;
    std::vector<double> expected(nlist * dim, 0.0), sums;
    std::vector<int64_t> expected_counts(nlist, 0), counts;
    auto serial = [&](std::vector<double>& s, std::vector<int64_t>& c) {
        s.assign(nlist * dim, 0.0);
        c.assign(nlist, 0);
        for (int64_t j = 0; j < nb; ++j) {
            ++c[owner[j]];
            for (int64_t d = 0; d < dim; ++d) s[owner[j] * dim + d] += x[j * dim + d];
        }
    };
    serial(expected, expected_counts);
    std::vector<double> times;
    for (int repeat = -2; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        if (grouped)
            knowhere::fl::AccumulateCentroids(x.data(), owner.data(), nb, dim, nlist, sums, counts);
        else
            serial(sums, counts);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (counts != expected_counts || std::memcmp(sums.data(), expected.data(), sums.size() * sizeof(double)) != 0)
            throw std::runtime_error("Centroid update differs bitwise from serial oracle");
        if (repeat >= 0)
            times.push_back(ms);
    }
    std::cout << "SAMPLES_MS";
    for (auto ms : times) std::cout << ' ' << ms;
    std::sort(times.begin(), times.end());
    std::cout << "\np50_ms=" << times[times.size() / 2] << " bitwise_equal=1\n";
}
