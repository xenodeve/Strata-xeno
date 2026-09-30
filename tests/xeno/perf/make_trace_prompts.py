"""Training prompts for routing traces, disjoint from the four benchmark prompts (code/thai/sky/long).

Wrapped like the benchmark prompt.ids: <|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
"""
import os, sys
sys.path.insert(0, r"D:\Github\Strata\.worktrees\q2-kernel\tools")
from strata_tokenizer import Tokenizer

GGUF = r"C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace-prompts")
HEAD = [248045, 846, 198]
TAIL = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]

PROMPTS = {
    "py-async": "Refactor this Python function to use asyncio and aiohttp, add type hints and error handling, "
                "and explain each change:\n\nimport requests\n\ndef fetch_all(urls):\n    out = []\n    for u in urls:\n"
                "        r = requests.get(u, timeout=10)\n        out.append(r.json())\n    return out\n",
    "thai-net": "อธิบายความแตกต่างระหว่าง TCP กับ UDP อย่างละเอียด พร้อมยกตัวอย่างการใช้งานจริงในระบบเกม วิดีโอสตรีมมิ่ง "
                "และระบบธนาคาร และบอกด้วยว่าเมื่อไรควรเลือกใช้แบบไหน",
    "bash-files": "Write a robust bash script that finds the 20 largest files under a given directory, skips "
                  "node_modules and .git, prints human-readable sizes, and exits non-zero on bad input. Then explain it.",
    "json-review": "Review this JSON configuration for a web service, explain every field, and point out security "
                   "and reliability problems:\n{\"port\": 8080, \"debug\": true, \"db\": {\"host\": \"0.0.0.0\", "
                   "\"user\": \"root\", \"password\": \"admin\"}, \"cors\": \"*\", \"timeout_ms\": 0, "
                   "\"workers\": 64, \"log_level\": \"trace\"}",
    "cpp-bug": "This C++ code sometimes crashes. Find the bug and fix it:\n\n#include <vector>\n#include <thread>\n"
               "std::vector<int> v;\nvoid add(int n){ for(int i=0;i<n;++i) v.push_back(i); }\nint main(){\n"
               "  std::thread a(add, 100000), b(add, 100000);\n  a.join(); b.join();\n  return (int) v.size();\n}\n",
    "thai-plan": "ช่วยวางแผนการย้ายระบบ monolith ที่เขียนด้วย PHP ไปเป็น microservices บน Kubernetes "
                 "แบ่งเป็นขั้นตอน ระบุความเสี่ยง และวิธีทดสอบในแต่ละขั้น",
}


def main():
    tk = Tokenizer.from_gguf(GGUF)
    os.makedirs(OUT, exist_ok=True)
    for name, text in PROMPTS.items():
        ids = HEAD + tk.encode(text) + TAIL
        open(os.path.join(OUT, name + ".ids"), "w").write(",".join(map(str, ids)))
        print(name, len(ids))


if __name__ == "__main__":
    main()
