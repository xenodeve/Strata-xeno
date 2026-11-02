// The fence and the session, on real tokens of packs/gyro-m when the path is given.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "strata/core/igc_sandbox_bridge.hpp"

using namespace strata::igc::sandbox;

static void require(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    std::exit(1);
}
static std::optional<std::string> feed(Fence& fence, std::vector<std::string> pieces) {
    std::optional<std::string> out;
    for (const auto& piece : pieces) {
        auto program = fence.step(piece);
        if (program) out = program;
    }
    return out;
}

int main(int argc, char** argv) {
    {
        Fence f;
        auto p = feed(f, {"\n\n", "```", "python", "\n", "print", "(", "1", ")", "\n", "```"});
        require(p && *p == "\nprint(1)\n", "a block closes at its fence");
        require(!f.inside(), "and leaves the block");
    }
    {
        Fence f;
        require(!feed(f, {"```", "\n", "F=(8/3,2)", "\n", "```", "\n"}), "a bare fence never runs");
        require(!feed(f, {"```", "text", "\n", "x", "\n", "```"}), "another language never runs");
        require(!f.inside(), "and stays outside");
    }
    {
        Fence f;
        require(!feed(f, {"```python\nprint(1)"}), "no closing fence yet");
        require(f.inside(), "inside");
        auto p = feed(f, {"\n", "``", "`"});
        require(p && *p == "\nprint(1)\n", "a fence split across tokens");
    }
    {
        Fence f;
        auto p = feed(f, {" ", "```py\n", "x = '```'\n", "```"});
        require(p && *p == "\nx = '```'\n", "backticks inside a line stay code");
    }
    {
        Fence f;
        auto a = feed(f, {"```python\n", "```"});
        require(a && *a == "\n", "an empty block");
        auto b = feed(f, {"\n", "```python\n", "print(2)\n", "```"});
        require(b && *b == "\nprint(2)\n", "a second block");
    }
    {
        Fence f;
        f.read_drafts(true);
        auto p = feed(f, {"Now:\n", "```", "\n", "class Solution:\n    def f(self, a):\n        b = a\n        return b\n", "```"});
        require(p && f.runs(*p), "a bare fence with a whole draft runs");
        auto q = feed(f, {"\n", "```", "\n", "F=(8/3,2)\n", "```"});
        require(q && !f.runs(*q), "a bare fence with no draft does not");
        auto r = feed(f, {"\n", "```cpp\n", "int main() {}\n", "```"});
        require(r && !f.runs(*r), "nor another language");
        require(!f.inside(), "and its closing fence opens nothing");
        require(!feed(f, {"Ensure delimiters: ", "```python ...```", ". Done.\n"}), "a fence in a sentence opens nothing");
        require(!f.inside(), "and leaves the state alone");
        auto s = feed(f, {"\n", "```", "python", "\n", "x = 1\n", "```"});
        require(s && f.runs(*s), "a python block split across tokens still runs");
        auto t = feed(f, {"\n", "``", "`\n", "def f(a):\n    b = a\n    c = b\n    return c\n", "```"});
        require(t && f.runs(*t), "a bare fence split across tokens");
    }
    {
        Fence f;
        f.enter();
        f.leave();
        require(!f.inside(), "a stopped block is left behind");
        require(!feed(f, {"\n", "```", "\n"}), "and its fence is not read as a close");
    }
    {
        Fence f;
        f.enter();
        require(f.inside(), "a forced fence counts as read");
        auto p = feed(f, {"print", "(", "3", ")", "\n", "```"});
        require(p && *p == "\nprint(3)\n", "and the block closes as any other");
    }
    {
        Fence f;
        f.enter(kCallHead);
        auto p = feed(f, {"f", "(", "5", "))", "\n", "```"});
        require(p && *p == "\nprint(f(5))\n", "a forced call keeps its head");
        require(binds_a_name("def f(n):\n    return n\n") && binds_a_name("x = 2\n"), "a block with names");
        require(last_def("\ndef a(x):\n    def inner():\n        pass\n    return 1\n\ndef main(n: int) -> int:\n    return a(n)\n") == "main", "the last top-level def");
        require(call_head("def f(n):\n    return n\n") == "print(f(", "the call names it");
        require(call_head("x = 2\n") == "print(", "no function, a bare print");
        require(call_head("class Solution:\n    def __init__(self):\n        pass\n    def helper(self, a):\n        return a\n    def solve(self, a: int) -> int:\n        return a\n") == "print(Solution().solve(", "a class's last method");
        require(call_head("class A:\n    def __init__(self):\n        pass\n") == "print(", "a class with no method to call");
        require(!binds_a_name("\n") && !binds_a_name("import math\n"), "a block with none has nothing to call");
    }
    {
        Session s;
        require(s.run("def f(n):\n    return n + 1\n") == output_block(kNothingPrinted) && s.quiet(), "a draft is quiet");
        require(s.run("print(f(1))\n") == "\n```output\n2\n```\n\n" && s.printed() && !s.quiet(), "its call prints");
        require(s.run("print()\n") == output_block(kNothingPrinted) && !s.printed() && s.quiet(), "an empty line says nothing");
        require(s.run("print(None)\n") == "\n```output\nNone\n```\n\n" && !s.printed() && !s.quiet(), "None is shown, but it is no result");
        require(s.run("print(None)\nprint(None)\n").find("None") != std::string::npos && !s.printed(), "nor are two of them");
        require(s.run("print(None)\nprint(1)\n").find("1") != std::string::npos && s.printed(), "a value after one is");
    }
    {
        Session s;
        const auto empty = s.run("import sys\ndef main():\n    d = sys.stdin.read().split()\n    print(sum(map(int, d)) if d else None)\nd = sys.stdin.read()\n");
        require(empty.find("stdin was empty") != std::string::npos && !s.quiet() && !s.printed(), "a program that read nothing says why, and no call is forced");
        require(s.run("feed(\"3 4\\n\")\nmain()\n") == "\n```output\n7\n```\n\n" && s.printed(), "fed, the same program runs");
    }
    {
        Session s;
        require(s.run("expect(1 + 1, 2)\n") == "\n```output\nmatch: 2\n```\n\n" && s.printed() && !s.mismatch(), "a right result says so");
        require(s.run("expect(1, 3)\n") == "\n```output\nMISMATCH: expected 3, got 1\n```\n\n" && s.mismatch(), "a wrong one says what was expected");
        require(s.run("expect(None, [1])\n").find("MISMATCH: expected [1], got None") != std::string::npos && s.mismatch(), "an unfinished draft is a mismatch");
        require(s.run("expect(0.1 + 0.2, 0.3)\n").find("match: ") != std::string::npos && !s.mismatch(), "floats within the benchmark's tolerance match");
        require(s.run("expect(1 + 1 == 2)\n").find("match: True") != std::string::npos && s.matched(), "one argument is a condition");
        require(s.run("expect(1 + 1 == 3)\n").find("MISMATCH: expected True, got False") != std::string::npos && s.mismatch() && !s.matched(), "a condition that fails");
        require(s.run("expect(2, 2)\nexpect(2, 3)\n").find("MISMATCH") != std::string::npos && !s.matched(), "one failure among passes is no pass");
        require(s.run("print(2)\n") == "\n```output\n2\n```\n\n" && !s.matched(), "a print is no check");
        require(s.run("print(1)\n") == "\n```output\n1\n```\n\n" && !s.mismatch(), "and the next run starts clean");
        require(call_head("def f(a):\n    return a\n", true).ends_with("expect(f("), "the forced call can be a test");
        require(about_the_oracle("\ndef brute(a):\n    return a\n") && about_the_oracle("\nexpect(brute(1), 2)\n"), "the oracle and its test");
        require(!about_the_oracle("\nclass Solution:\n    def f(self, a):\n        return a\n") && !about_the_oracle("\nexpect(f(1), brute(1))\n"), "not the draft, nor the draft set against it");
        require(call_head("def f(a):\n    return a\n").starts_with("print(f("), "and stays a print by default");
    }
    {
        Session s;
        int calls = 0;
        std::string sent;
        bool first = false;
        s.set_exec([&](std::string_view code, bool fresh, std::string& body, bool& ok) {
            if (!calls) first = fresh;
            ++calls;
            sent = std::string(code);
            body = "2\n";
            ok = true;
            return true;
        });
        require(s.run("x = 1\nprint(x)\n") == "\n```output\n1\n```\n\n" && calls == 0, "what the interpreter can do stays there");
        require(s.run("import sympy\nprint(sympy.Integer(2))\n") == "\n```output\n2\n```\n\n" && calls == 1 && first && s.printed(), "a refusal goes to real Python");
        require(sent.starts_with("x = 1\n"), "with the blocks before it");
        s.run("print(x)\n");
        require(calls == 2 && sent == "print(x)\n", "and the turn stays there");
    }
    {
        Session s;
        require(s.run("x = input()\n").find("stdin was empty") != std::string::npos && !s.printed(), "input() past the end says why");
        require(s.run("feed('a b\\nc\\n')\nprint(input().split(), input())\n") == "\n```output\n['a', 'b'] c\n```\n\n", "input() reads what was fed");
        require(s.run("print(g(1))\n").find("NameError") != std::string::npos && !s.quiet() && !s.printed(), "an error is neither");
    }
    {
        Session s;
        require(s.run("x = 2\nprint(x)\n") == "\n```output\n2\n```\n\n", "first block");
        require(s.run("print(x + 1)\n") == "\n```output\n3\n```\n\n", "names carry over and old prints go");
        require(s.printed(), "it printed");
        require(s.run("y = 5\n") == output_block(kNothingPrinted), "nothing printed");
        require(!s.printed(), "and says so");
        const auto failed = s.run("print(7)\nprint(z)\n");
        require(failed.find("7\n") != std::string::npos && failed.find("NameError") != std::string::npos, "a print, then the error");
        require(s.run("print(y)\n") == "\n```output\n5\n```\n\n", "a failed block leaves no trace");
        require(s.runs() == 5, "runs counted");
        require(s.seen("print(y)\n") == 1 && s.seen("print(9)\n") == 0, "a repeated program is seen");
    }
    {
        Session s;
        s.run("a = 1\nb = 2\nc = 3\n");
        const auto failed = s.run("\nx = 1\nassert x == 2\n");
        require(failed.find("AssertionError (line 2)\n    assert x == 2\n") != std::string::npos, "the line in this block, and its text");
        s.run("def f(n):\n    return n[0]\n");
        const auto deeper = s.run("f([])\n");
        require(deeper.find("(in an earlier block)\n    return n[0]") != std::string::npos, "a failure inside an earlier block says so");
        const auto syntax = s.run("x = = 1\n");
        require(syntax.find("(line 1)\n    x = = 1") != std::string::npos, "a syntax error too");
        require(name_the_line("TypeError", "x\n", 0) == "TypeError", "no line, nothing to name");
        require(name_the_line("ValueError (line 9)", "x\n", 0) == "ValueError (line 9)", "a line past the source");
    }
    {
        Session s;
        const auto long_output = s.run("for i in range(2000):\n    print(i)\n");
        require(long_output.find("[output cut]") != std::string::npos, "a long output is cut");
        require(long_output.size() < kShownOutput + 64, "to the shown size");
    }
    {
        Session s;
        require(s.run("while True:\n    pass\n").find("```output\n") != std::string::npos, "a loop stops at the step limit");
    }
    {
        Session s;
        require(s.run("print('สวัสดี')\n") == "\n```output\nสวัสดี\n```\n\n", "text outside ASCII");
        const auto refused = s.run("import os\n");
        require(refused.find("os") != std::string::npos && refused.find("```output\n") == 1, "an unsupported import reports");
        require(refused.find(kOnlyModules) != std::string::npos, "and says what is here");
        require(!s.printed(), "a refused import does not count as a result");
        require(s.run("x = 1 / 0\n").find(kOnlyModules) == std::string::npos, "other errors stay as they are");
        require(s.run("print(1)\n") == "\n```output\n1\n```\n\n", "and the session goes on");
    }
    TokenText small;
    std::string error;
    require(!small.parse("{\"a\": 1}", error), "a vocabulary with missing bytes is refused");
    if (argc > 1) {
        TokenText vocab;
        require(vocab.load(argv[1], error), error.c_str());
        std::string opened;
        for (const int32_t t : kOpenBlock) opened += vocab.text(t);
        require(opened == "Need a short Python program here.\n\n```python\n", "the forced text");
        require(vocab.text(198) == "\n", "newline");
        require(vocab.text(220) == " ", "space");
        require(vocab.text(71093) == "```", "fence");
        require(vocab.text(12305) == "python", "python");
        require(vocab.text(52451) == " ```", "spaced fence");
        require(vocab.text(271) == "\n\n", "blank line");
        require(vocab.text(9999999).empty(), "no such token");
        const auto block = vocab.encode(output_block("12.0"));
        require(std::find(block.begin(), block.end(), 71093) != block.end(), "a fence is one token");
        require(std::find(block.begin(), block.end(), 63) == block.end(), "and no single backticks");
        const auto call = vocab.encode(std::string(kBlockOpen) + call_head("def f(n):\n    return n\n"));
        require(call.size() >= 4 && call[0] == 71093 && call[1] == 12305 && call[2] == 198, "the forced call opens as the forced block does");
        std::string back;
        for (const int32_t t : vocab.encode("สวัสดี 12")) back += vocab.text(t);
        require(back == "สวัสดี 12", "bytes come back");
    }
    std::puts("PASS igc sandbox bridge");
    return 0;
}
