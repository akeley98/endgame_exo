"""Flag SASS register moves that copy out of an LDSM destination quad.

Heuristic: LDSM.*.4 Rn writes Rn..Rn+3. Any MOV / IMAD.MOV reading one of those
registers (before it is redefined) means ptxas had to shuffle ldmatrix output
to satisfy a consumer's register-vector order (e.g. HMMA wants an aligned pair).

Straight-line within each function; good enough for unrolled kernels and loop bodies.
"""

import re
import sys

ins_re = re.compile(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s+([^;]*);")
reg_re = re.compile(r"\bR(\d+)\b")


def check(sass: str):
    report = {}
    for func in re.split(r"\n\s*Function : ", sass)[1:]:
        name = func.split("\n")[0].strip()
        live = {}  # register -> address of LDSM that wrote it
        n_ldsm = 0
        bad = []
        for m in ins_re.finditer(func):
            op, operands = m.group(1), m.group(2)
            regs = [int(r) for r in reg_re.findall(operands)]
            if not regs:
                continue
            dst, srcs = regs[0], regs[1:]
            if op.startswith(("MOV", "IMAD.MOV")):
                hit = [r for r in srcs if r in live]
                if hit:
                    bad.append(f"{op} {operands}")
            if op.startswith("LDSM"):
                n_ldsm += 1
                nreg = 4 if op.endswith(".4") else 2 if op.endswith(".2") else 1
                for r in range(dst, dst + nreg):
                    live[r] = True
            elif not op.startswith(("HMMA", "ST", "RED", "ATOM", "BAR", "BRA", "EXIT")):
                live.pop(dst, None)  # first register is the destination
            elif op.startswith("HMMA"):
                live.pop(dst, None)
                live.pop(dst + 1, None)
        report[name] = (n_ldsm, bad)
    return report


if __name__ == "__main__":
    rc = 0
    for path in sys.argv[1:]:
        for name, (n_ldsm, bad) in check(open(path).read()).items():
            status = "OK " if not bad else "BAD"
            print(f"{status} {path}: {name}: {n_ldsm} LDSM, {len(bad)} shuffles")
            for b in bad[:4]:
                print("      ", b)
            rc |= bool(bad)
    sys.exit(rc)
