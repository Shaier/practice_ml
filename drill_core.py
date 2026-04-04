import ast
import os
import random
import uuid
from typing import Dict, List, Optional

_COMMENT_PREFIXES = ("#", '"""', "'''", '"', "'", "`")
_STRUCTURAL_PREFIXES = ("class ",)

EXERCISES: Dict[str, dict] = {}


def read_file(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        return f.readlines()


def get_top_level_items(filepath: str) -> List[dict]:
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    items = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            items.append({
                "name": node.name,
                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                "start": node.lineno - 1,
                "end": node.end_lineno,
            })
    return items


def get_docstring_line_indices(filepath: str) -> set:
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                ds = node.body[0]
                for i in range(ds.lineno - 1, ds.end_lineno):
                    docstring_lines.add(i)
    return docstring_lines


def get_maskable_indices(all_lines: List[str], start: int, end: int, docstring_lines: set) -> List[int]:
    maskable = []
    for i in range(start + 1, end):
        if i in docstring_lines:
            continue
        stripped = all_lines[i].strip()
        if not stripped:
            continue
        if stripped.startswith(_COMMENT_PREFIXES):
            continue
        if stripped.startswith(_STRUCTURAL_PREFIXES):
            continue
        maskable.append(i)
    return maskable


def scan_drills(drills_dir: str = "drills") -> List[dict]:
    items = []
    if not os.path.exists(drills_dir):
        return items
    for fname in sorted(os.listdir(drills_dir)):
        if not fname.endswith(".py"):
            continue
        filepath = os.path.join(drills_dir, fname)
        try:
            file_items = get_top_level_items(filepath)
            for item in file_items:
                items.append({
                    "file": filepath,
                    "name": item["name"],
                    "type": item["type"],
                    "label": f"{fname}::{item['name']}",
                })
        except SyntaxError:
            pass
    return items


def generate_exercise(target_label: Optional[str], mask_value: float, guidance: bool) -> dict:
    drill_items = scan_drills()
    if not drill_items:
        raise ValueError("No drillable items found in drills/")

    if target_label:
        pool = [x for x in drill_items if x["label"] == target_label]
        if not pool:
            raise ValueError(f"Item '{target_label}' not found")
    else:
        pool = drill_items

    chosen = random.choice(pool)
    filepath = chosen["file"]
    item_name = chosen["name"]

    all_lines = read_file(filepath)
    doc_lines = get_docstring_line_indices(filepath)
    file_items = get_top_level_items(filepath)
    item = next(x for x in file_items if x["name"] == item_name)
    start, end = item["start"], item["end"]

    maskable = get_maskable_indices(all_lines, start, end, doc_lines)
    if not maskable:
        raise ValueError(f"No maskable lines in '{item_name}'")

    if mask_value <= 1.0:
        mask_count = max(1, round(len(maskable) * mask_value))
    else:
        mask_count = int(mask_value)

    to_mask = set(random.sample(maskable, min(mask_count, len(maskable))))

    lines = []
    answers = {}
    bid = 0

    for abs_i, raw_line in enumerate(all_lines[start:end], start=start):
        line_text = raw_line.rstrip("\n")
        display_n = abs_i - start + 1
        if abs_i in to_mask:
            indent = len(line_text) - len(line_text.lstrip())
            answers[str(bid)] = line_text.strip()
            lines.append({
                "n": display_n,
                "blank": True,
                "bid": bid,
                "indent": line_text[:indent],
                "placeholder": "# code here" if guidance else "",
            })
            bid += 1
        else:
            lines.append({"n": display_n, "blank": False, "text": line_text})

    exercise_id = str(uuid.uuid4())[:8]
    EXERCISES[exercise_id] = {
        "id": exercise_id,
        "source": f"{os.path.relpath(filepath)}::{item_name}",
        "lines": lines,
        "num_blanks": bid,
        "_answers": answers,
    }

    return {k: v for k, v in EXERCISES[exercise_id].items() if k != "_answers"}


def evaluate_exercise(exercise_id: str, answers: Dict[str, str]) -> dict:
    if exercise_id not in EXERCISES:
        raise ValueError("Exercise not found — generate a new one")
    ex = EXERCISES[exercise_id]
    correct = ex["_answers"]
    results = []
    n_correct = 0
    for bid_str, expected in correct.items():
        user = answers.get(bid_str, "").strip()
        ok = user == expected.strip()
        if ok:
            n_correct += 1
        results.append({"bid": int(bid_str), "correct": ok, "yours": user, "answer": expected})
    return {
        "score": f"{n_correct}/{len(correct)}",
        "all_correct": n_correct == len(correct),
        "results": results,
    }