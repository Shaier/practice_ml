import ast
import os
import random
import argparse
import sys


def read_file(filepath):
    with open(filepath, "r") as f:
        return f.readlines()


def get_top_level_items(filepath):
    """Return only top-level classes and standalone functions (not methods inside classes)."""
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    items = []
    for node in tree.body:  # tree.body = top-level nodes only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            items.append({
                "name": node.name,
                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                "start": node.lineno - 1,   # 0-indexed, inclusive
                "end": node.end_lineno,      # 0-indexed, exclusive
            })
    return items


def get_docstring_line_indices(filepath):
    """Return set of 0-indexed line numbers that belong to any docstring in the file."""
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


_COMMENT_PREFIXES = ("#", '"""', "'''", '"', "'", "`")
_STRUCTURAL_PREFIXES = ("def ", "async def ", "class ")


def get_maskable_indices(all_lines, start, end, docstring_lines):
    """Return line indices that are meaningful to mask.
    Skips: def/class lines (including nested), blank lines, comments,
           docstrings, and bare string/backtick lines.
    """
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


def apply_mask(all_lines, maskable_indices, mask_count, guidance):
    """Return modified lines with mask_count randomly chosen lines masked."""
    lines = list(all_lines)
    to_mask = random.sample(maskable_indices, min(mask_count, len(maskable_indices)))
    for idx in sorted(to_mask):
        if guidance:
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            lines[idx] = " " * indent + "# code here\n"
        else:
            lines[idx] = "\n"
    return lines, len(to_mask)


def generate_todo(source_filepath, item, all_lines, docstring_lines, mask_value, guidance, output_path):
    start = item["start"]
    end = item["end"]

    maskable = get_maskable_indices(all_lines, start, end, docstring_lines)
    if not maskable:
        print(f"  Warning: no maskable lines found in '{item['name']}', skipping.")
        return 0

    if mask_value < 1:
        mask_count = max(1, round(len(maskable) * mask_value))
    else:
        mask_count = int(mask_value)

    masked_lines, n_masked = apply_mask(all_lines, maskable, mask_count, guidance)

    func_lines = masked_lines[start:end]

    rel_path = os.path.relpath(source_filepath)
    with open(output_path, "w") as f:
        f.write(f"# source: {rel_path}::{item['name']}\n")
        f.write(f"# masked: {n_masked} line(s)\n")
        f.write("\n")
        f.writelines(func_lines)
        if not func_lines[-1].endswith("\n"):
            f.write("\n")

    return n_masked


def next_todo_number(todo_dir):
    existing = [
        f for f in os.listdir(todo_dir)
        if f.startswith("todo_") and f.endswith(".py")
    ]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f[len("todo_"):-len(".py")]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate fill-in-the-blank TODO files from your drills/ folder."
    )
    parser.add_argument("--file", type=str, help="Specific .py file inside drills/ to use")
    parser.add_argument("--function", type=str, help="Specific function or class name to drill")
    parser.add_argument("--num", type=int, default=1, help="Number of TODO files to generate (default: 1)")
    parser.add_argument(
        "--mask", type=float, required=True,
        help="Masking amount: <1 = fraction of lines (0.5 = 50%%), >=1 = exact line count"
    )
    parser.add_argument(
        "--guidance", action="store_true",
        help="Replace masked lines with '# code here' instead of blank lines"
    )
    args = parser.parse_args()

    drills_dir = "drills"
    if not os.path.exists(drills_dir):
        print(f"Error: '{drills_dir}/' directory not found. Create it and add your .py files.")
        sys.exit(1)

    # collect python files
    if args.file:
        candidate = args.file if os.path.isabs(args.file) else os.path.join(drills_dir, args.file)
        if not os.path.exists(candidate):
            print(f"Error: '{candidate}' not found.")
            sys.exit(1)
        py_files = [candidate]
    else:
        py_files = [
            os.path.join(drills_dir, f)
            for f in os.listdir(drills_dir)
            if f.endswith(".py")
        ]

    if not py_files:
        print(f"No .py files found in '{drills_dir}/'. Add some code to drill on.")
        sys.exit(1)

    # parse all files and collect items
    pool = []  # list of (filepath, item, all_lines, docstring_lines)
    for filepath in py_files:
        try:
            all_lines = read_file(filepath)
            items = get_top_level_items(filepath)
            docstring_lines = get_docstring_line_indices(filepath)
            for item in items:
                pool.append((filepath, item, all_lines, docstring_lines))
        except SyntaxError as e:
            print(f"Warning: skipping '{filepath}' (syntax error: {e})")

    if not pool:
        print("No functions or classes found in drills/.")
        sys.exit(1)

    # filter by function/class name if specified
    if args.function:
        filtered = [(fp, it, ln, ds) for fp, it, ln, ds in pool if it["name"] == args.function]
        if not filtered:
            available = sorted(set(it["name"] for _, it, _, _ in pool))
            print(f"Error: '{args.function}' not found.")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        pool = filtered

    os.makedirs("todo_exercises", exist_ok=True)
    start_n = next_todo_number("todo_exercises")

    for i in range(args.num):
        filepath, item, all_lines, docstring_lines = random.choice(pool)
        output_path = f"todo_exercises/todo_{start_n + i}.py"
        n_masked = generate_todo(filepath, item, all_lines, docstring_lines, args.mask, args.guidance, output_path)
        if n_masked:
            rel = os.path.relpath(filepath)
            print(f"  {output_path}  ←  {item['type']} '{item['name']}' from {rel}  ({n_masked} lines masked)")


if __name__ == "__main__":
    main()