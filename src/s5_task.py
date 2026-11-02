"""S5 box-swap state-tracking task: generation, prompt text, parsing, classification.

Pure Python (CPU only). Every operation is a transposition, i.e. a bijection on
states, so editing the state at step m ALWAYS changes the final answer (y' != y).
"""
import json
import math
import random
import re

NBOX = 4                      # v3: 4 boxes (pilot v2: 1-2B models at floor with 5)
LETTERS = "ABCDE"[:NBOX]
POS = range(1, NBOX + 1)

# ----------------------------------------------------------------------------- core dynamics


def apply_swap(state, op):
    i, j = op
    s = list(state)
    s[i - 1], s[j - 1] = s[j - 1], s[i - 1]
    return "".join(s)


def run(init, ops):
    out, s = [], init
    for op in ops:
        s = apply_swap(s, op)
        out.append(s)
    return out


def rand_ops(rng, k):
    ops, prev = [], None
    while len(ops) < k:
        i, j = sorted(rng.sample(list(POS), 2))
        if (i, j) == prev:          # avoid immediate self-cancelling swaps
            continue
        ops.append((i, j))
        prev = (i, j)
    return ops


def make_problem(rng, k, pid):
    init = "".join(rng.sample(LETTERS, NBOX))
    ops = rand_ops(rng, k)
    states = run(init, ops)                       # states[t-1] = state after step t
    m = math.ceil(k / 2)                          # edit position (middle step)
    a, b = rng.sample(range(NBOX), 2)
    s = list(states[m - 1])
    s[a], s[b] = s[b], s[a]
    edit_m = "".join(s)
    edit_traj = [edit_m] + run(edit_m, ops[m:])   # states for steps m..k under the edit
    assert edit_m != states[m - 1]
    assert edit_traj[-1] != states[-1], "bijectivity violated"
    return dict(pid=pid, k=k, m=m, init=init, ops=json.dumps(ops),
                states=json.dumps(states), edit_state=edit_m,
                edit_traj=json.dumps(edit_traj), y=states[-1], y_edit=edit_traj[-1])


# ----------------------------------------------------------------------------- prompt text


def fmt(state):
    """Position-labelled state, e.g. '1:C 2:B 3:A 4:D'."""
    return " ".join(f"{i}:{c}" for i, c in enumerate(state, 1))


_DESC = ("Four boxes are numbered 1 to 4 and each box holds one letter. A state is written as "
         "'1:C 2:B 3:A 4:D', meaning box 1 holds C, box 2 holds B, and so on. "
         "The operation 'swap X and Y' exchanges the letters in box X and box Y; all other boxes stay the same.\n")
INSTR_COT = _DESC + (
    "Apply the operations in order. For each operation write 'Step n (swap X and Y):' followed by the "
    "state after that operation. Finally write 'Answer:' followed by the final state.\n")
INSTR_DIRECT = _DESC + (
    "Apply the operations in order, but do NOT write any intermediate steps. "
    "Write only 'Answer:' followed by the final state.\n")


def problem_block(init, ops):
    lines = [f"Initial: {fmt(init)}", "Operations:"]
    lines += [f"{t}. swap {i} and {j}" for t, (i, j) in enumerate(ops, 1)]
    return "\n".join(lines) + "\n"


def steps_text(states, start=1, ops=None):
    """ops[t] is the operation that produced states[t]."""
    out = []
    for t, s in enumerate(states):
        tag = f" (swap {ops[t][0]} and {ops[t][1]})" if ops is not None else ""
        out.append(f"Step {start + t}{tag}: {fmt(s)}\n")
    return "".join(out)


def _examples():
    rng = random.Random(7)
    exs = []
    for kk in (2, 3, 4):
        init = "".join(rng.sample(LETTERS, NBOX))
        ops = rand_ops(rng, kk)
        exs.append((init, ops, run(init, ops)))
    return exs


_EXAMPLES = _examples()


def examples_text(mode):
    out = []
    for n, (init, ops, states) in enumerate(_EXAMPLES, 1):
        sol = steps_text(states, ops=ops) if mode == "cot" else ""
        out.append(f"Example {n}:\n{problem_block(init, ops)}Solution:\n{sol}Answer: {fmt(states[-1])}\n")
    return "\n".join(out)


def user_text(mode, init, ops):
    instr = INSTR_COT if mode == "cot" else INSTR_DIRECT
    return f"{instr}\n{examples_text(mode)}\nNow solve this problem.\n{problem_block(init, ops)}"


def build_items(problems):
    """Model-agnostic items: user text + assistant prefix + gold completion."""
    rows = []
    for p in problems.to_dict("records"):
        ops = [tuple(o) for o in json.loads(p["ops"])]
        states = json.loads(p["states"])
        et = json.loads(p["edit_traj"])
        k, m = p["k"], p["m"]
        base = dict(pid=p["pid"], k=k, m=m)
        rows.append(dict(base, cond="cot", user_text=user_text("cot", p["init"], ops),
                         assistant_prefix="Solution:\n",
                         gold_completion=steps_text(states, ops=ops) + f"Answer: {fmt(p['y'])}\n"))
        rows.append(dict(base, cond="direct", user_text=user_text("direct", p["init"], ops),
                         assistant_prefix="Solution:\n",
                         gold_completion=f"Answer: {fmt(p['y'])}\n"))
        prefix = ("Solution:\n" + steps_text(states[: m - 1], ops=ops[: m - 1])
                  + f"Step {m} (swap {ops[m-1][0]} and {ops[m-1][1]}): {fmt(p['edit_state'])}\n")
        rows.append(dict(base, cond="edit", user_text=user_text("cot", p["init"], ops),
                         assistant_prefix=prefix,
                         gold_completion=steps_text(et[1:], start=m + 1, ops=ops[m:]) + f"Answer: {fmt(p['y_edit'])}\n"))
        rows.append(dict(base, cond="fresh", user_text=user_text("cot", p["edit_state"], ops[m:]),
                         assistant_prefix="Solution:\n",
                         gold_completion=steps_text(et[1:], ops=ops[m:]) + f"Answer: {fmt(p['y_edit'])}\n"))
    import pandas as pd
    df = pd.DataFrame(rows)
    df["item_id"] = df["pid"] + "|" + df["cond"]
    return df


# ----------------------------------------------------------------------------- parsing

_STEP_LINE = re.compile(r"(?i)\bstep\s*(\d+)[^:\n]*:(.*)")
_ANS_LINE = re.compile(r"(?i)\banswer\b\s*[:\-]?(.*)")
_LABEL = re.compile(r"\b\d\s*[:=.)]\s*")


def _state_from(rest):
    """'1:C 2:B 3:A 4:D' or 'C B A D' -> 'CBAD' (first NBOX letters), else None."""
    rest = _LABEL.sub(" ", rest.replace("*", " "))
    toks = re.findall(r"\b[A-Z]\b", rest)[:NBOX]
    return "".join(toks) if len(toks) == NBOX else None


def _valid(s):
    return s is not None and len(s) == NBOX and set(s) == set(LETTERS)


def parse_output(text):
    """Return dict(steps={n: state}, answer=state|None, n_steps=int). Invalid states are dropped."""
    steps, ans = {}, None
    for line in (text or "").splitlines():
        mt = _STEP_LINE.search(line)
        if mt:
            st = _state_from(mt.group(2))
            if _valid(st):
                steps[int(mt.group(1))] = st
            continue
        mt = _ANS_LINE.search(line)
        if mt:
            st = _state_from(mt.group(1))
            ans = st if _valid(st) else None
            break                                  # first answer line (generation stops there)
    return dict(steps=steps, answer=ans, n_steps=len(steps))


def classify_edit(parsed, prob):
    """Outcome class of an edited trace (see paper Sec. Method)."""
    ans, k = parsed["answer"], prob["k"]
    y, y_edit = prob["y"], prob["y_edit"]
    if ans is None:
        return "invalid"
    if ans == y_edit:
        return "follows_edit"
    if ans == y:
        last = parsed["steps"].get(k)
        if last == y:
            return "repair"              # trace visibly returned to the original path
        if last == y_edit:
            return "dissociation"        # trace follows edit, answer does not
        return "retained_unexplained"
    return "wrong_other"


def first_post_edit_path(parsed, prob):
    m = prob["m"]
    s = parsed["steps"].get(m + 1)
    if s is None:
        return "none"
    states = json.loads(prob["states"])
    et = json.loads(prob["edit_traj"])
    if s == et[1]:
        return "edit_path"
    if s == states[m]:
        return "orig_path"
    return "other"
