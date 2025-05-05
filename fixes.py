# %% ──────────────────────────────────────────────────────────────────────────
# Greedy decode + quick visual sanity‑check
import os, torch, matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

device = torch.device("cpu")        # or "cuda"
model  = model.to(device).eval()

# ────────────────────────────────────────────────────────────────────────────
def preprocess(img, patch_embed):
    """img : (1,56,56) → (1,seq,D)"""
    return patch_embed(img.unsqueeze(0))

def greedy_decode(model, img, patch_embed, tokenizer, max_len=8):
    """
    img        : (1,56,56) tensor, values in [-1,1]
    patch_embed: model.embed   (patchifier)
    returns list of *clean* digit tokens  (no specials)
    """
    mem = model.encoder(preprocess(img, patch_embed))  # (1,S,D)

    start_id  = tokenizer.token_to_idx["<start>"]
    finish_id = tokenizer.token_to_idx["<finish>"]
    pad_id    = tokenizer.token_to_idx["<pad>"]

    ys = torch.tensor([[start_id, pad_id]], device=img.device)  # seed <start> <pad>

    for step in range(max_len - 1):                 # already have 2 slots
        logits = model.decoder(ys, mem)             # (1,t,vocab)  t ≥ 1

        # Prevent predicting <finish> right after <start>
        # This logic might need adjustment based on when exactly we want to prevent it.
        # If we only want to prevent it on the *very first* prediction step:
        if ys.shape[1] == 2: # If sequence length is exactly 2 (<start>, <pad>)
             logits[:, -1, finish_id] = -float('inf')
        # If we want to prevent it anytime before a non-special token is generated,
        # the logic would be more complex. Sticking to the simple case for now.

        next_id = logits[:, -1].argmax(-1)          # (1,)
        ys[0, -1] = next_id                         # overwrite the dummy pad

        if next_id.item() == finish_id:
            break                                   # finished -> stop expanding

        # Append a *new* pad token for the next prediction step
        ys = torch.cat([ys, torch.tensor([[pad_id]], device=img.device)], dim=1)

    # strip specials
    clean = [i for i in ys[0].tolist()
             if i not in {start_id, finish_id, pad_id}]
    return [tokenizer.idx_to_token[i] for i in clean]

# ────────────────────────────────────────────────────────────────────────────
# Quick demo on first 50 test samples + optional PNG dump
def dump_preds(n=50, out_dir="./model_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    print("🔍 Dumping model predictions…")

    for idx in range(n):
        sample = test_grid[idx]
        img    = sample['image'].to(device)
        gold   = [VOCAB_INV[i] for i in sample['target'].tolist()
                  if i not in {PAD_IDX, VOCAB['<start>'], VOCAB['<finish>']}]

        pred   = greedy_decode(model, img, model.embed, tokenizer, max_len=10)

        print(f"[{idx:03d}]  True: {' '.join(gold)}")
        print(f"      Pred: {' '.join(pred)}")

        png = TF.to_pil_image(img.squeeze().cpu()*0.5 + 0.5)
        fname = f"{idx:03d}_gold={'-'.join(gold)}_pred={'-'.join(pred)}.png"
        png.save(os.path.join(out_dir, fname))

    print("✅ Done.")

# — run the dump —
dump_preds(n=50)
