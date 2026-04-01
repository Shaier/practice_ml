# ML drills

Fill-in-the-blank coding drills for learning ML implementations.

## setup

Put your Python implementations in `drills/`. Any top-level class or function becomes drillable.

```
transformer_drills/
  drills/
    attention.py
    transformer_block.py
    ...              ← add your own files here too
  drill.py
  drill_core.py
  eval_work.py
```

## running

```bash
python drill.py
```

That's it. A browser window opens automatically at `http://localhost:7234`.

## using the UI

**Sidebar (left panel)**

| Control | What it does |
|---|---|
| Exercise | Pick a specific file + function/class, or leave on *Random* |
| Mask % | Percentage of code lines to blank out |
| Guidance hints | When on, blanks show `# code here`; when off, lines are completely empty |
| Generate | Creates a new exercise with the current settings |

**Editor (main panel)**

- Blank lines have a blue left border and an inline input field
- Indentation is preserved — just type the code
- **Tab** / **Shift+Tab** moves between blanks
- **Ctrl+Enter** (or the Evaluate button) checks your answers

**Results**

- Correct blanks turn green
- Wrong blanks turn red with a strikethrough, and the correct answer appears on the line below — right in context
- Score shows in the sidebar

## example for what's in drills/

| File | Contents |
|---|---|
| `attention.py` | Masks, scaled dot-product attention, MHA, cross-attention |
| `attention_variants.py` | Multi-query attention, grouped-query attention, KV cache |
| `positional.py` | Sinusoidal encoding, learned embeddings, RoPE |
| `normalization.py` | LayerNorm, RMSNorm, BatchNorm |
| `feedforward.py` | FFN, GatedFFN (SwiGLU / GeGLU) |
| `transformer_block.py` | Pre-norm block, post-norm block, decoder block |
| `architectures.py` | TransformerEncoder, TransformerDecoder, EncoderDecoder, GPT, BERT |
| `recurrent.py` | RNNCell, LSTMCell, RNN, LSTM |
| `losses_and_activations.py` | Softmax, swish, GELU, SiLU, cross-entropy, focal loss |
| `moe.py` | Switch MoE (top-1), Soft MoE (top-k) |
| `contrastive.py` | InfoNCE, NT-Xent (SimCLR), CLIP loss, dual encoder |
| `convolutions.py` | Residual block, simple CNN (Conv2d), TCN (Conv1d + dilation) |
| `classical_ml.py` | Linear regression, logistic regression, precision/recall/F1, k-means |
| `distributed.py` | Data parallelism, model parallelism, pipeline parallelism |
| `generative.py` | Autoencoder, VAE, GAN, straight-through estimator, VQ-VAE |


## suggested workflow

1. Start with 5-10% mask and guidance on — learn the structure
2. Increase to 20–50% once the skeleton feels familiar
3. Turn guidance off — now you have to remember what's missing
4. Hit 100% mask on individual functions for blank-slate practice
5. Repeat the same function after 3 days, 1 week, 2 weeks