import ast
import os
import sys
import argparse


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.readlines()


def parse_metadata(lines):
    """Extract source filepath and function/class name from the metadata header."""
    source_file = None
    func_name = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# source:"):
            rest = stripped[len("# source:"):].strip()
            if "::" in rest:
                source_file, func_name = rest.split("::", 1)
                source_file = source_file.strip()
                func_name = func_name.strip()
    return source_file, func_name


def get_original_lines(filepath, func_name):
    """Extract the source lines of a specific function/class from a file."""
    with open(filepath, "r") as f:
        source = f.read()
    all_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == func_name:
                return all_lines[node.lineno - 1 : node.end_lineno]
    return None


def get_user_lines(all_lines):
    """Strip the metadata header lines and return just the user's code."""
    result = []
    past_header = False
    for line in all_lines:
        stripped = line.strip()
        if not past_header:
            if (
                stripped.startswith("# source:")
                or stripped.startswith("# masked:")
                or stripped == ""
            ):
                continue
            else:
                past_header = True
        result.append(line)
    return result


def compare(original_lines, user_lines):
    """Line-by-line comparison. Returns list of result dicts."""
    orig = [l.rstrip() for l in original_lines]
    user = [l.rstrip() for l in user_lines]

    results = []
    for i in range(max(len(orig), len(user))):
        o = orig[i] if i < len(orig) else None
        u = user[i] if i < len(user) else None

        if o is None:
            results.append({"line": i + 1, "status": "extra", "yours": u, "correct": None})
        elif u is None:
            results.append({"line": i + 1, "status": "missing", "yours": None, "correct": o})
        elif o.strip() == u.strip():
            results.append({"line": i + 1, "status": "ok", "yours": u, "correct": o})
        else:
            u_stripped = u.strip()
            if u_stripped == "" or u_stripped == "# code here":
                status = "blank"
            else:
                status = "wrong"
            results.append({"line": i + 1, "status": status, "yours": u, "correct": o})

    return results


def build_report(todo_path, source_file, func_name, results, show_answers):
    """Build the full report as a string."""
    total = len(results)
    correct = sum(1 for r in results if r["status"] == "ok")
    issues = [r for r in results if r["status"] != "ok"]
    bar = "─" * 52

    lines_out = []
    lines_out.append(bar)
    lines_out.append(f"  file:   {todo_path}")
    lines_out.append(f"  source: {source_file} :: {func_name}")

    if not issues:
        lines_out.append(f"  score:  {correct}/{total}  ✓  perfect!")
    else:
        lines_out.append(f"  score:  {correct}/{total}  —  {len(issues)} line(s) need work")
        lines_out.append("")
        for r in issues:
            ln = r["line"]
            if r["status"] == "blank":
                lines_out.append(f"  line {ln:>3}:  ✗  not filled in")
            elif r["status"] == "wrong":
                lines_out.append(f"  line {ln:>3}:  ✗  yours:   {r['yours'].strip()!r}")
            elif r["status"] == "missing":
                lines_out.append(f"  line {ln:>3}:  ✗  line is missing")
            elif r["status"] == "extra":
                lines_out.append(f"  line {ln:>3}:  ✗  unexpected: {r['yours'].strip()!r}")

            if show_answers and r["correct"] is not None:
                lines_out.append(f"           ✓  correct: {r['correct'].strip()!r}")

    lines_out.append(bar)
    return "\n".join(lines_out) + "\n"


def results_path_for(todo_path):
    """Given todo_exercises/todo_3.py, return todo_exercises/todo_3_results.txt"""
    base = os.path.splitext(todo_path)[0]
    return base + "_results.txt"


def eval_file(todo_path, show_answers):
    try:
        lines = read_file(todo_path)
    except FileNotFoundError:
        print(f"error: '{todo_path}' not found.")
        return False

    source_file, func_name = parse_metadata(lines)
    if not source_file or not func_name:
        print(f"error: {todo_path}: missing or malformed '# source:' metadata comment.")
        return False

    if not os.path.exists(source_file):
        print(f"error: {todo_path}: source file '{source_file}' not found.")
        return False

    original_lines = get_original_lines(source_file, func_name)
    if original_lines is None:
        print(f"error: {todo_path}: could not find '{func_name}' in '{source_file}'.")
        return False

    user_lines = get_user_lines(lines)
    results = compare(original_lines, user_lines)

    total = len(results)
    correct = sum(1 for r in results if r["status"] == "ok")
    passed = correct == total

    report = build_report(todo_path, source_file, func_name, results, show_answers)

    out_path = results_path_for(todo_path)
    with open(out_path, "w") as f:
        f.write(report)

    status = "✓" if passed else f"{total - correct} issue(s)"
    print(f"  {todo_path}  →  {correct}/{total}  {status}  (see {out_path})")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate your filled-in TODO drill files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific todo file(s) to evaluate. Defaults to all files in todo_exercises/",
    )
    parser.add_argument(
        "--show-answers",
        action="store_true",
        help="Write the correct line next to each wrong answer in the results file",
    )
    args = parser.parse_args()

    if args.files:
        todo_files = args.files
    else:
        todo_dir = "todo_exercises"
        if not os.path.exists(todo_dir):
            print("No 'todo_exercises/' directory found. Run drill.py first.")
            sys.exit(1)
        todo_files = sorted(
            os.path.join(todo_dir, f)
            for f in os.listdir(todo_dir)
            if f.startswith("todo_") and f.endswith(".py")
        )

    if not todo_files:
        print("No TODO files found. Run drill.py to generate some.")
        sys.exit(1)

    all_passed = True
    for path in todo_files:
        passed = eval_file(path, args.show_answers)
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  all exercises passed!")


if __name__ == "__main__":
    main()