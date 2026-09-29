#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) return 2;
    std::vector<char> records(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) records.data(), &bytes)) return 2;
    std::vector<BYTE> classes;
    std::vector<int> core_ids;
    for (const char* p = records.data(), *end = records.data() + bytes; p < end;) {
        const auto* r = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (r->Relationship == RelationProcessorCore) {
            const auto& mask = r->Processor.GroupMask[0];
            for (int bit = 0; bit < 64; ++bit)
                if (mask.Mask & (1ull << bit)) {
                    classes.push_back(r->Processor.EfficiencyClass);
                    core_ids.push_back((int) (mask.Group * 64 + bit));
                    break;
                }
        }
        p += r->Size;
    }
    if (classes.size() < 2) return 2;
    const BYTE highest = *std::max_element(classes.begin(), classes.end());
    const int preferred = highest == 0 ? (int) classes.size() :
                          (int) std::count(classes.begin(), classes.end(), highest);
    const int p_only = std::max(1, preferred - 1); // first high-class core reserved for the host
    const int expected_auto = (int) classes.size() - 1;
    const auto all = strata::kernels::cpu::physical_cores(false);
    const auto workers = strata::kernels::cpu::physical_cores(true);
    bool ordered = all.size() == classes.size() && workers.size() + 1 == all.size();
    for (int i = 0; ordered && i < (int) all.size(); ++i) {
        const auto it = std::find(core_ids.begin(), core_ids.end(), all[(size_t) i]);
        if (it == core_ids.end()) { ordered = false; break; }
        const BYTE cls = classes[(size_t) (it - core_ids.begin())];
        if ((i < preferred && cls != highest) || (i >= preferred && highest != 0 && cls == highest))
            ordered = false;
    }
    for (int i = 0; ordered && i < (int) workers.size(); ++i)
        ordered = workers[(size_t) i] == all[(size_t) i + 1];
    strata::kernels::cpu::ExpertPool automatic(0, false, true);
    const int actual = automatic.workers();
    strata::kernels::cpu::ExpertPool p_cores_only(p_only, false, true);
    strata::kernels::cpu::ExpertPool with_e_cores(p_only + 1, false, true);
    const bool vnni = strata::kernels::cpu::cpu_avxvnni_ok();
    std::printf("physical %zu, highest class %u, preferred %d, auto workers %d/%d, P-only %d\n",
                classes.size(), (unsigned) highest, preferred, actual, expected_auto, p_cores_only.workers());
    std::printf("native Q2_0 dispatch: AVX-VNNI %s\n", vnni ? "yes" : "no");
    bool dispatch = true;
    if (argc == 2 && std::strcmp(argv[1], "--expect-vnni") == 0)
        dispatch = vnni && !strata::kernels::cpu::cpu_avx512_ok();
    else if (argc == 2 && std::strcmp(argv[1], "--expect-avx2") == 0)
        dispatch = !vnni && !strata::kernels::cpu::cpu_avx512_ok();
    else if (argc != 1) return 2;
    return ordered && actual == expected_auto && p_cores_only.workers() == p_only &&
           with_e_cores.workers() == p_only + 1 && dispatch ? 0 : 1;
}
