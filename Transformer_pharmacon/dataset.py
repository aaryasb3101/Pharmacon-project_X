import pandas as pd
import torch
from torch.utils.data import Dataset


class ParentMetaboliteDataset(Dataset): #create dataset class that inherits from pytorch's dataset base class
    """
    Dataset for paired parent-drug and metabolite SMILES.

    CSV columns used:
        Precursor SMILES -> parent drug
        SMILES           -> metabolite
    """

    def __init__(
        self,
        csv_path,
        tokenizer,
        max_len=256,
    ):
        self.data = pd.read_csv(csv_path) #load dataset to pandas df

        self.tokenizer = tokenizer
        self.max_len = max_len

        # Check required columns
        required_columns = [
            "Precursor SMILES",
            "SMILES",
        ]

        for column in required_columns: #loops through required coloumns one at a time and checks if it exists in dataframe
            if column not in self.data.columns:
                raise ValueError(
                    f"Missing required column: {column}" #if the coloumn does not exist, stops and tells us exactly which coloumn is missing
                )

        # Remove rows with missing SMILES
        self.data = self.data.dropna(
            subset=[
                "Precursor SMILES",
                "SMILES",
            ]
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.data) #checks how many samples are in the dataset

    def __getitem__(self, idx): #basically tells pytorch to get the sample no. idx

        row = self.data.iloc[idx] #iloc is integer location indexing

        parent_smiles = row["Precursor SMILES"]
        metabolite_smiles = row["SMILES"]

        # Tokenize parent + metabolite
        parent = self.tokenizer.encode_batch(
            [parent_smiles],
            max_len=self.max_len,
        )

        metabolite = self.tokenizer.encode_batch(
            [metabolite_smiles],
            max_len=self.max_len,
        )

        return {
            "parent_smiles": parent_smiles,
            "metabolite_smiles": metabolite_smiles,

            "parent_input_ids":
                parent["input_ids"].squeeze(0),

            "parent_attention_mask":
                parent["attention_mask"].squeeze(0),

            "metabolite_input_ids":
                metabolite["input_ids"].squeeze(0),

            "metabolite_attention_mask":
                metabolite["attention_mask"].squeeze(0),
        }