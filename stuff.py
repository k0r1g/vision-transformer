import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

grid_transforms = transforms.Compose([
    transforms.ToTensor(),                  # uint8 → 0-1
    transforms.Normalize([0.5], [0.5])      # centre to −1..1
])

class GridMNIST(Dataset):
    """
    Creates a 56×56 canvas divided into 16 implicit 14×14 cells.
      • A random number (1‑16) of cells are filled.
      • Digits are sampled with replacement; class frequencies are uniform.
      • Target sequence = <start>  odd↑  even↓  <finish>
    """
    def __init__(self, base_images, base_labels,
                 epoch_size=60_000, rng=None):
        self.base_images = base_images          # (N,28,28) uint8
        self.base_labels = base_labels          # (N,)
        self.epoch_size  = epoch_size
        self.rng = np.random.default_rng(rng)

        # Store indices for each digit 0-9 for quick lookup
        self.per_digit = {d: np.where(base_labels == d)[0] for d in range(10)}

    def __len__(self):
        # Return the desired size of the dataset for one epoch
        return self.epoch_size

    def _random_cells(self):
        # Decide how many cells to fill (1 to 16)
        n = self.rng.integers(1, 17)
        # Choose 'n' unique cell IDs randomly from 0 to 15
        return self.rng.choice(16, size=n, replace=False)

    def __getitem__(self, idx):
        # Get a random set of cell IDs for this specific item
        cell_ids = self._random_cells()

        # Create a blank canvas (56x56 pixels, black)
        # Assumes CANVAS_PIX is defined, e.g., CANVAS_PIX = 56
        canvas = np.zeros((CANVAS_PIX, CANVAS_PIX), dtype=np.uint8)
        digits = [] # To store the digits we place

        # Loop through each chosen cell ID
        for cell in cell_ids:
            # Pick a random digit (0-9)
            d = int(self.rng.integers(0, 10))
            # Find an index in the original dataset for this digit
            img_idx   = self.rng.choice(self.per_digit[d])
            # Get the actual 28x28 image of the digit
            digit_img = self.base_images[img_idx]

            # Resize the 28x28 digit image to fit in a cell (e.g., 14x14)
            # Assumes CELL_PIX is defined, e.g., CELL_PIX = 14
            # We need PIL's Image and torchvision's functional transforms (TF)
            digit_img = TF.resize(Image.fromarray(digit_img), CELL_PIX)
            # Convert back to a numpy array
            digit_img = np.array(digit_img, dtype=np.uint8)

            # Figure out where this cell is on the 4x4 grid
            # Assumes GRID_SIZE is defined, e.g., GRID_SIZE = 4
            row, col = divmod(cell, GRID_SIZE)
            # Calculate the top-left pixel coordinate for this cell
            top, left = row*CELL_PIX, col*CELL_PIX
            # Place the resized digit image onto the canvas
            canvas[top:top+CELL_PIX, left:left+CELL_PIX] = digit_img

            # Add the digit value (0-9) to our list
            digits.append(d)

        # --- Build the target sequence ---
        # Get all odd digits and sort them in increasing order
        odds  = sorted([d for d in digits if d % 2 == 1])
        # Get all even digits and sort them in decreasing order
        evens = sorted([d for d in digits if d % 2 == 0], reverse=True)
        # Combine them: start token, odds, evens, finish token
        # Assumes VOCAB dictionary is defined with '<start>' and '<finish>' keys
        seq   = [VOCAB['<start>']] + odds + evens + [VOCAB['<finish>']]

        # Return the final image and target sequence
        return {
            # Apply transformations (like ToTensor, Normalize) to the image
            # Assumes grid_transforms is defined elsewhere
            'image'  : grid_transforms(canvas),
            # Convert the sequence list into a PyTorch tensor
            'target' : torch.tensor(seq, dtype=torch.long)
        }

# Collate function to add padding to the targets 
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    images  = torch.stack([item['image'] for item in batch])  # shape: (B, 1, 56, 56)
    targets = [item['target'] for item in batch]              # list of length-B tensors
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)  # (B, T_max)
    return {'image': images, 'target': padded_targets}



# ─── build train / val / test loaders ───────────────────────────────────────
train_grid = GridMNIST(train_dataset.data.numpy(),
                       train_dataset.targets.numpy(),
                       epoch_size=60_000, rng=RANDOM_SEED)

val_grid   = GridMNIST(test_dataset.data.numpy(),   # we reuse MNIST test set
                       test_dataset.targets.numpy(),
                       epoch_size=10_000, rng=RANDOM_SEED+1)

test_grid  = GridMNIST(test_dataset.data.numpy(),
                       test_dataset.targets.numpy(),
                       epoch_size=10_000, rng=RANDOM_SEED+2)

# ——————————————————————————————————————————————————————————————
# Decide how many CPU cores to devote to data loading.
# Four to eight usually keeps the GPU fed without wasting resources.
# ——————————————————————————————————————————————————————————————
NUM_WORKERS = min(8, os.cpu_count())        # 4-8 is a good starting range

loader_kwargs = dict(
    batch_size      = BATCH_SIZE,
    num_workers     = NUM_WORKERS,          # <- key addition
    pin_memory      = True,                 # speeds up host-to-device copy
    persistent_workers = True,              # keeps workers alive across epochs
    prefetch_factor = 4                     # each worker holds 4 batches ready
)

train_dataloader = DataLoader(
    train_grid,
    shuffle=True,
    collate_fn=collate_batch,       
    **loader_kwargs
)

val_dataloader = DataLoader(
    val_grid,
    shuffle=False,
    collate_fn=collate_batch,           
    **loader_kwargs
)

test_dataloader = DataLoader(
    test_grid,
    shuffle=False,
    collate_fn=collate_batch,      
    **loader_kwargs
)

