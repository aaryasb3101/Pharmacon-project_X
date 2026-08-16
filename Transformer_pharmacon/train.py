import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tokenizer import SmilesTokenizer
from dataset import ParentMetaboliteDataset
from model import SmilesTransformer
from pretrain import ParentMetaboliteContrastiveLoss


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"

MAX_LEN = 256


# ============================================================
# 2-LAYER TRANSFORMER
# ============================================================

D_MODEL = 768
NUM_HEADS = 8
NUM_LAYERS = 2
D_FF = 3072


# ============================================================
# REGULARIZATION
# ============================================================

DROPOUT = 0.3
WEIGHT_DECAY = 1e-4


# ============================================================
# TRAINING
# ============================================================

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8

EPOCHS = 26
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07

# Increased from 5 → 10
PATIENCE = 10

SAVE_DIR = "checkpoints"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# LOAD DATA
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Training records:", len(train_df))
print("Validation records:", len(val_df))

# TOKENIZER
tokenizer = SmilesTokenizer()

train_smiles = (
    train_df["Precursor SMILES"]
    .astype(str)
    .tolist()
    +
    train_df["SMILES"]
    .astype(str)
    .tolist()
)

tokenizer.build_vocab(train_smiles)

print("Vocabulary size:", tokenizer.vocab_size)
print("PAD ID:", tokenizer.pad_id)
print("CLS ID:", tokenizer.cls_id)
print("UNK ID:", tokenizer.unk_id)

# DATASETS
train_dataset = ParentMetaboliteDataset(
    csv_path=TRAIN_CSV,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_dataset = ParentMetaboliteDataset(
    csv_path=VAL_CSV,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# MODEL
model = SmilesTransformer(
    vocab_size=tokenizer.vocab_size,
    pad_id=tokenizer.pad_id,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT
)

model = model.to(device)

print("Transformer layers:", NUM_LAYERS)
print("Attention heads:", NUM_HEADS)
print("Dropout:", DROPOUT)
print("Weight decay:", WEIGHT_DECAY)
print("Learning rate:", LEARNING_RATE)
print("Early stopping patience:", PATIENCE)

# LOSS
criterion = ParentMetaboliteContrastiveLoss(
    temperature=TEMPERATURE
)

# OPTIMIZER
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# CHECKPOINT
os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

best_val_loss = float("inf")
epochs_without_improvement = 0

# NEW CHECKPOINT — DOES NOT OVERWRITE THE PREVIOUS RUN
best_path = os.path.join(
    SAVE_DIR,
    "smiles_transformer_2layer_8head_dropout03_patience10_best.pt"
)

# LOSS HISTORY
train_losses = []
val_losses = []

# TRAINING LOOP
for epoch in range(EPOCHS):

    # TRAINING
    model.train()

    total_train_loss = 0.0

    for batch in train_loader:

        parent_ids = batch[
            "parent_input_ids"
        ].to(device)

        parent_mask = batch[
            "parent_attention_mask"
        ].to(device)

        metabolite_ids = batch[
            "metabolite_input_ids"
        ].to(device)

        metabolite_mask = batch[
            "metabolite_attention_mask"
        ].to(device)

        optimizer.zero_grad()

        parent_embedding, _, _ = model(
            parent_ids,
            parent_mask
        )

        metabolite_embedding, _, _ = model(
            metabolite_ids,
            metabolite_mask
        )

        loss = criterion(
            parent_embedding,
            metabolite_embedding
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    train_loss = (
        total_train_loss / len(train_loader)
    )

    train_losses.append(train_loss)

    # VALIDATION
    # Dropout intentionally remains ON.
    # Gradients remain OFF.
    model.train()

    total_val_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            parent_ids = batch[
                "parent_input_ids"
            ].to(device)

            parent_mask = batch[
                "parent_attention_mask"
            ].to(device)

            metabolite_ids = batch[
                "metabolite_input_ids"
            ].to(device)

            metabolite_mask = batch[
                "metabolite_attention_mask"
            ].to(device)

            parent_embedding, _, _ = model(
                parent_ids,
                parent_mask
            )

            metabolite_embedding, _, _ = model(
                metabolite_ids,
                metabolite_mask
            )

            loss = criterion(
                parent_embedding,
                metabolite_embedding
            )

            total_val_loss += loss.item()

    val_loss = (
        total_val_loss / len(val_loader)
    )

    val_losses.append(val_loss)

    # PRINT
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )

    # SAVE BEST MODEL
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "vocab_size":
                    tokenizer.vocab_size,

                "num_layers":
                    NUM_LAYERS,

                "num_heads":
                    NUM_HEADS,

                "d_model":
                    D_MODEL,

                "d_ff":
                    D_FF,

                "dropout":
                    DROPOUT,

                "weight_decay":
                    WEIGHT_DECAY,

                "learning_rate":
                    LEARNING_RATE,

                "train_batch_size":
                    TRAIN_BATCH_SIZE,

                "val_batch_size":
                    VAL_BATCH_SIZE,

                "patience":
                    PATIENCE,

                "dropout_during_validation":
                    True
            },
            best_path
        )

        print(
            f"  → Best 2-layer model saved "
            f"(val loss: {val_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  → No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    # EARLY STOPPING
    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping triggered "
            f"at epoch {epoch + 1}."
        )

        break

# SAVE LOSS HISTORY
history_df = pd.DataFrame({
    "epoch": list(
        range(
            1,
            len(train_losses) + 1
        )
    ),
    "train_loss": train_losses,
    "val_loss": val_losses
})

history_path = os.path.join(
    SAVE_DIR,
    "2layer_8head_dropout03_patience10_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

# COMPLETE
print("\nTraining complete!")

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Best model:",
    best_path
)

print(
    "Loss history:",
    history_path
)