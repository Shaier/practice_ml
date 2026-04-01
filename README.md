# transformer drills

Fill-in-the-blank coding drills.

## setup

Put your Python files in `drills/`. Any functions or classes in there become drillable.

```
drills/
  attention.py       ← your MHA, sdp_attention, etc.
  transformer.py     ← your TransformerBlock, etc.
```

## generate drills

```bash
# 1 drill, mask 40% of lines, with guidance hints
python drill.py --mask 0.4 --guidance

# 5 drills from a specific file
python drill.py --mask 0.4 --num 5 --file attention.py

# drill a specific function, mask exactly 3 lines
python drill.py --mask 3 --function MultiHeadAttention

# hard mode: no hints, mask 60%
python drill.py --mask 0.6 --num 3
```

**`--mask`**: if < 1, treats as a fraction (0.5 = 50% of lines). If >= 1, masks that many lines exactly.  
**`--guidance`**: masked lines show `# code here`. Without it, lines are just blank.

Generated files land in `todo_exercises/todo_1.py`, `todo_2.py`, etc.

## do the drill

Open a todo file, fill in the blanks. The top of each file tells you where it came from:

```python
# source: drills/attention.py::MultiHeadAttention
# masked: 4 line(s)
```

## check your work

```bash
# check all todo files
python eval_work.py

# check a specific file
python eval_work.py todo_exercises/todo_1.py

# show correct answers next to wrong lines
python eval_work.py --show-answers
```

## workflow

1. Add your implementations to `drills/`
2. Run `drill.py` to generate a TODO
3. Fill in the blanks (no peeking)
4. Run `eval_work.py --show-answers` to check
5. Repeat after 3 days, 1 week, 2 weeks