import torch
import pandas as pd
from torch.utils.data import DataLoader

from tokenizer import SmilesTokenizer
from dataset import ParentMetaboliteDataset
from model import SmilesTransformer
from pretrain import ParentMetaboliteContrastiveLoss

# Configuration
CSV_PATH = "final_metabolites.csv"
MAX_LEN = 256

D_MODEL = 768
NUM_HEADS = 12
NUM_LAYERS = 12
D_FF = 3072
DROPOUT = 0.1

BATCH_SIZE = 2



# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# Load CSV
df = pd.read_csv(CSV_PATH)

print("\nDataset size:", len(df))

# Build Tokenizer
tokenizer = SmilesTokenizer()

all_smiles = (
    df["Precursor SMILES"].astype(str).tolist()
    + df["SMILES"].astype(str).tolist()
)

tokenizer.build_vocab(all_smiles)

print("Vocabulary size:", tokenizer.vocab_size)

# Create Dataset
dataset = ParentMetaboliteDataset(
    csv_path=CSV_PATH,
    tokenizer=tokenizer,
    max_len=MAX_LEN,
)

print("Dataset samples:", len(dataset))

# Create Small DataLoader
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Get One Batch
batch = next(iter(loader))

print("\n--- Batch Shapes ---")

print(
    "Parent input:",
    batch["parent_input_ids"].shape
)

print(
    "Parent mask:",
    batch["parent_attention_mask"].shape
)

print(
    "Metabolite input:",
    batch["metabolite_input_ids"].shape
)

print(
    "Metabolite mask:",
    batch["metabolite_attention_mask"].shape
)


# Create Model
model = SmilesTransformer(
    vocab_size=tokenizer.vocab_size,
    pad_id=tokenizer.pad_id,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN,
    dropout=DROPOUT,
)

model = model.to(device)

print("\nModel created successfully.")

# Move Batch to Device
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

# Forward Pass
model.train()

parent_embedding, _, _ = model(
    parent_ids,
    parent_mask,
)

metabolite_embedding, _, _ = model(
    metabolite_ids,
    metabolite_mask,
)


print("\n--- Embedding Shapes ---")

print(
    "Parent embedding:",
    parent_embedding.shape
)

print(
    "Metabolite embedding:",
    metabolite_embedding.shape
)

# Loss
criterion = ParentMetaboliteContrastiveLoss(
    temperature=0.07
)

loss = criterion(
    parent_embedding,
    metabolite_embedding,
)

print(
    "\nContrastive loss:",
    loss.item()
)

# Backward Pass
loss.backward()

print("\nBackward pass successful.")

# Check Gradients
gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        gradient_found = True

        print(
            "Gradient found:",
            name
        )

        break


if gradient_found:
    print("\n SANITY TEST PASSED!")
    print("Tokenizer → Dataset → Model → Loss → Backpropagation")
    print("Everything is connected correctly.")

else:
    print("\n No gradients found.")