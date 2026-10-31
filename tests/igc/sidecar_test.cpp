// Parity of the engine's sidecar with the trained Python model, and its time per token.
// Input is a text file of cases, written from the fixture of tools/igc/sidecar_export.py.
#include "strata/core/igc_sidecar.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: sidecar_test <weights> <cases.txt>\n"; return 2; }
    strata::igc::Sidecar sidecar;
    std::string error;
    if (!sidecar.load(argv[1], error)) { std::cerr << error << "\n"; return 1; }
    std::ifstream in(argv[2]);
    int cases = 0;
    in >> cases;
    double worst = 0;
    for (int c = 0; c < cases; ++c) {
        int length = 0, entries = 0;
        in >> length >> entries;
        std::vector<int> tokens(length), ends(entries);
        std::vector<float> operation(5), left(entries), right(entries), got_o(5), got_l(entries), got_r(entries);
        for (int& v : tokens) in >> v;
        for (int& v : ends) in >> v;
        for (float& v : operation) in >> v;
        for (float& v : left) in >> v;
        for (float& v : right) in >> v;
        sidecar.reset();
        const auto start = std::chrono::steady_clock::now();
        for (int token : tokens) sidecar.push(token);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        sidecar.score(ends, got_o.data(), got_l.data(), got_r.data());
        double most = 0;
        for (int i = 0; i < 5; ++i) most = std::max(most, std::abs(static_cast<double>(got_o[i] - operation[i])));
        for (int i = 0; i < entries; ++i) {
            most = std::max(most, std::abs(static_cast<double>(got_l[i] - left[i])));
            most = std::max(most, std::abs(static_cast<double>(got_r[i] - right[i])));
        }
        worst = std::max(worst, most);
        std::cout << length << " tokens " << ms << " ms, " << ms / length << " ms per token, largest difference " << most << "\n";
    }
    if (worst > 2e-2) { std::cerr << "FAILED: differs from the trained model\n"; return 1; }
    std::cout << "OK\n";
    return 0;
}
