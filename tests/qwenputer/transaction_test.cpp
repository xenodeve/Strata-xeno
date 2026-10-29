#include "strata/core/qwenputer_transaction.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace strata::qwenputer;
static int failures = 0;
static void check(bool condition, const char* name) { if (!condition) { ++failures; std::cerr << "FAIL " << name << '\n'; } }
static std::vector<std::uint8_t> read(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
static Digest state(const Executor& executor) {
    // Full mutable circuit S plus lag history; independent of transaction internals.
    std::vector<std::uint8_t> bytes((12845056 + 350160));
    check(cudaMemcpy(bytes.data(), executor.device_state(), 12845056, cudaMemcpyDeviceToHost) == cudaSuccess, "read S");
    check(cudaMemcpy(bytes.data()+12845056, executor.device_lags(), 350160, cudaMemcpyDeviceToHost) == cudaSuccess, "read lag");
    return sha256(bytes);
}
int main(int argc, char** argv) {
    const bool short_path=argc==5 && std::string_view(argv[4])=="--short";
    if (argc!=4 && !short_path) { std::cerr << "artifact binding prefix-inputs [--short]\n"; return 2; }
    const auto identity = read(argv[2]), prefix_raw = read(argv[3]);
    if (identity.size() != 192 || prefix_raw.size() % sizeof(Input)) return 2;
    Binding binding; std::memcpy(binding.digests.data(), identity.data(), 192);
    Artifact artifact; MemoryPlan memory; std::string error;
    if (!load_artifact(argv[1], binding, {}, artifact, memory, error)) { std::cerr << error; return 1; }
    std::vector<Input> prefix(prefix_raw.size()/sizeof(Input)); std::memcpy(prefix.data(), prefix_raw.data(), prefix_raw.size());
    if (prefix.size() < 7 || prefix.back().id != artifact.config[8]) return 2;
    std::vector<std::uint32_t> ids; for (const auto t:prefix) { if (t.kind!=TokenKind::Text) return 2; ids.push_back(t.id); }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 1;
    {
        Executor executor;
        if (!executor.init(artifact, stream, error)) { std::cerr << error; return 1; }
        {
            auto wrong = artifact; wrong.digest[0] ^= 1;
            std::size_t before=0, after=0, total=0;
            check(cudaMemGetInfo(&before,&total)==cudaSuccess,"memory before binding refusal");
            Transaction refused;
            check(!refused.init(executor,wrong,stream,{}, {},error),"wrong executor/artifact pair refused");
            check(cudaMemGetInfo(&after,&total)==cudaSuccess && before==after,"binding refusal precedes GPU allocation");
        }
        for (int mode=0; mode<2; ++mode) {
            std::size_t before=0, after=0, total=0;
            check(cudaMemGetInfo(&before,&total)==cudaSuccess,"memory before argument refusal");
            RuntimeLimits limits; if(mode==0)limits.chunk=0;
            Transaction refused;
            check(!refused.init(executor,artifact,mode==1?nullptr:stream,limits,{},error),"invalid limit or stream refused");
            check(cudaMemGetInfo(&after,&total)==cudaSuccess && before==after,"argument refusal precedes GPU allocation");
        }
        Transaction transaction;
        if (!transaction.init(executor, artifact, stream, {}, {}, error)) { std::cerr << "FAIL init: " << error << '\n'; return 1; }
        if (short_path) {
            const std::uint32_t token=static_cast<std::uint32_t>(artifact.printable['a']);
            WindowDecision decision;
            check(transaction.begin_assistant(0,true,error),"short begin");
            check(transaction.trial({&token,1},1,decision,error),"short trial");
            check(transaction.commit([](int count,std::string&){return count==1;},error),"short commit");
            const auto prior=state(executor);
            check(transaction.trial({&token,1},1,decision,error),"short discarded trial");
            check(transaction.abort_trial(error) && state(executor)==prior,"short rollback");
            check(transaction.trial({&token,1},1,decision,error),"short failing language commit trial");
            check(!transaction.commit([](int,std::string& e){e="injected_language_commit_failure";return false;},error)
                    && !transaction.valid() && error=="injected_language_commit_failure","failed language commit invalidates whole session");
        } else {
        for (int row = 0; row <= 4; ++row) {
            check(transaction.reset(error), "reset"); check(transaction.begin_assistant(0, true, error), "begin assistant");
            const auto before = ids.size() - static_cast<std::size_t>(row) - 1;
            WindowDecision decision;
            check(transaction.trial(std::span(ids).first(before), static_cast<int>(before), decision, error), "prepare captured body");
            int committed = -1;
            check(transaction.commit([&](int count, std::string&) { committed=count; return true; }, error), "commit body");
            check(committed == static_cast<int>(before), "language callback count");
            const auto committed_state = state(executor);
            std::vector<std::uint32_t> window(ids.begin()+static_cast<std::ptrdiff_t>(before), ids.end());
            window.push_back(static_cast<std::uint32_t>(artifact.printable['x']));
            check(transaction.trial(window, static_cast<int>(window.size()), decision, error), "trial closing row");
            check(decision.consumed_inputs == row+1 && decision.visible_model_outputs == row && decision.takeover_row == row, "exact input ownership boundary");
            check(transaction.abort_trial(error), "abort speculative candidate");
            check(state(executor) == committed_state && transaction.position() == before && !transaction.owned(), "rejected shadow leaves base intact");
            if (row > 0) {
                check(transaction.trial(window, row, decision, error), "closing row in rejected suffix");
                check(decision.consumed_inputs == row && decision.takeover_row == -1, "rejected takeover has no ownership");
                check(transaction.abort_trial(error), "abort rejected suffix control");
            }
            check(transaction.trial(window, static_cast<int>(window.size()), decision, error), "repeat same candidate");
            check(transaction.commit([&](int count, std::string&) { committed=count; return true; }, error), "fenced accepted replay");
            check(committed == row+1 && transaction.position() == ids.size() && transaction.owned(), "only accepted inputs committed");
            if (row == 4) {
                std::vector<std::uint32_t> result; RunReason reason;
                check(transaction.run_owned(nullptr, result, reason, error), "native owned burst");
                check(reason == RunReason::Success && !result.empty(), "success stages complete result");
                std::string text;
                for(auto id:result) { const auto raw=artifact.token_bytes(id); text.append(reinterpret_cast<const char*>(raw.data()),raw.size()); }
                check(text.find("391") != std::string::npos, "native multiplication result");
                const auto post_burst = state(executor);
                const auto steps = transaction.text_steps();
                check(transaction.trial(result, static_cast<int>(result.size()), decision, error), "known result text window");
                check(transaction.commit([](int, std::string&) { return true; }, error), "language result replay");
                check(transaction.replay_remaining() == 0 && transaction.text_steps() == steps && state(executor) == post_burst, "result text never enters circuit twice");
            }
        }
        // Commit a valid input before a rejected closing-call row. The shadow's
        // parser must not leave an armed ownership decision in committed state.
        check(transaction.reset(error),"reset rejected-commit control");
        check(transaction.begin_assistant(0,true,error),"begin rejected-commit control");
        WindowDecision decision;
        check(transaction.trial(std::span(ids).first(ids.size()-2),static_cast<int>(ids.size()-2),decision,error),"body before final newline");
        check(transaction.commit([](int,std::string&){return true;},error),"commit partial body");
        std::vector<std::uint32_t> tail{ids[ids.size()-2],ids.back(),static_cast<std::uint32_t>(artifact.printable['x'])};
        check(transaction.trial(tail,1,decision,error) && decision.takeover_row==-1,"TCC lies in rejected suffix");
        check(transaction.commit([](int count,std::string&){return count==1;},error),"commit only accepted newline");
        check(!transaction.owned() && transaction.position()==ids.size()-1,"rejected TCC did not enter committed parser");
        tail.erase(tail.begin());
        check(transaction.trial(tail,2,decision,error) && decision.takeover_row==0,"subsequent accepted TCC takes ownership");
        check(transaction.commit([](int count,std::string&){return count==1;},error) && transaction.owned(),"accepted TCC commits once");
        }
    }
    check(cudaStreamDestroy(stream)==cudaSuccess,"destroy stream");
    std::cout << (failures ? "FAIL " : "PASS ") << failures << " failures\n";
    return failures ? 1 : 0;
}
