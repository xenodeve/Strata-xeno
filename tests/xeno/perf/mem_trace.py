"""RAM over time for one strata run: the machine's in-use and committed memory and the process's private bytes and
working set every 0.5 s, with each stderr line time-stamped, so the boot peak and the steady state can be read
from the same run.

usage: python mem_trace.py OUT_NAME PROMPT exe [extra args...]
"""
import json, os, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab  # noqa: E402

def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 3:
        raise ValueError("usage: mem_trace.py OUT_NAME PROMPT exe [extra args...]")
    out_name, prompt, exe, extra = args[0], args[1], args[2], args[3:]
    out = os.path.join(ab.T, out_name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(ab.T, ab.REF[prompt], "on.command.json")) as source:
        cmd = json.load(source)[1:] + ["--pool-priority", "2"] + extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
    cuda = os.path.join("C:" + os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v13.3", "bin")
    env["PATH"] = os.pathsep.join([cuda, os.path.join(cuda, "x64"), env["PATH"]])
    t0 = time.time()
    pr = subprocess.Popen([exe] + cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          encoding="utf-8", errors="replace", creationflags=0x80 | 0x08000000)
    samples, events, errors = [], [], []

    def receive(row):
        samples.append((time.time() - t0, row[0], row[2], row[3], row[4]))

    def read_err():
        for line in pr.stderr:
            events.append((time.time() - t0, line.rstrip()))

    threads = [threading.Thread(target=ab.collect_memory, args=(pr, receive, errors), daemon=True),
               threading.Thread(target=read_err, daemon=True)]
    for thread in threads: thread.start()
    stdout = pr.stdout.read()
    pr.wait()
    for thread in threads: thread.join()
    with open(os.path.join(out, "trace.tsv"), "w", encoding="utf-8") as f:
        f.write("t_s\tsys_in_use_gib\tsys_commit_gib\tproc_private_gib\tproc_ws_gib\n")
        for sample in samples: f.write("\t".join(f"{v:.2f}" for v in sample) + "\n")
    with open(os.path.join(out, "events.txt"), "w", encoding="utf-8") as f:
        for t, line in events: f.write(f"{t:8.2f}  {line}\n")
    with open(os.path.join(out, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(stdout)
    if errors or not samples:
        with open(os.path.join(out, "memory-errors.json"), "w", encoding="utf-8") as f:
            json.dump(dict(errors=errors, samples=len(samples), source="GetPerformanceInfo"), f, indent=2)
        raise RuntimeError("memory telemetry incomplete; raw logs preserved, see memory-errors.json")
    peak = max(samples, key=lambda sample: sample[1])
    print(f"rc={pr.returncode} samples={len(samples)} peak in-use {peak[1]:.2f} GiB at {peak[0]:.1f}s")


if __name__ == "__main__":
    main()
