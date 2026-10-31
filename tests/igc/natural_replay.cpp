// Replays written answers through the real firer: every firing on text the model already got right is a false one.
// Input lines: `case prompt-ids | generated-ids`. Output lines: `case position written replaced confidence`.
#include "strata/core/igc_natural.hpp"
#include "strata/core/igc_sidecar.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    strata::igc::Sidecar sidecar;
    std::string error;
    if (!sidecar.load(argv[1], error)) { std::cerr << error << "\n"; return 1; }
    std::ifstream input(argv[2]);
    const float threshold = argc > 3 ? std::stof(argv[3]) : .5f;
    for (std::string line; std::getline(input, line);) {
        std::istringstream row(line);
        std::string name, word;
        row >> name;
        std::vector<int64_t> prompt;
        std::vector<int32_t> generated;
        bool after = false;
        while (row >> word) {
            if (word == "|") after = true;
            else if (after) generated.push_back(std::stoi(word));
            else prompt.push_back(std::stoll(word));
        }
        strata::igc::NaturalFirer<strata::igc::Sidecar> firer(sidecar, threshold);
        firer.begin(prompt);
        for (size_t i = 0; i < generated.size(); ++i) {
            int32_t token = generated[i];
            const int64_t before = firer.fires();
            firer.step(token);
            if (firer.fires() != before) std::cout << name << ' ' << i << ' ' << generated[i] << ' ' << token << ' ' << firer.last_confidence() << '\n';
        }
    }
    return 0;
}
