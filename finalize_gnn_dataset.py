import pandas as pd
import numpy as np
from pathlib import Path

def finalize_gnn_dataset():
    script_dir = Path(__file__).resolve().parent
    enriched_path = script_dir / "ddis_final_enriched_dataset.csv"
    
    if not enriched_path.exists():
        print("Error: Run the enrichment script first.")
        return
        
    df = pd.read_csv(enriched_path)
    print(f"Loaded enriched DDI dataset with {len(df)} rows.")

    # 1. Fill gaps where DrugBank ID is missing using the original TWOSIDES ID
    # This ensures unmapped drugs are still included in the graph network structure.
    df["d1_drugbank_id"] = df["d1_drugbank_id"].fillna(df["d1"])
    df["d2_drugbank_id"] = df["d2_drugbank_id"].fillna(df["d2"])

    # 2. Extract a unique list of all drugs across the dataset
    all_drugs = pd.concat([df["d1"], df["d2"]]).unique()
    print(f"Total unique drug nodes in network: {len(all_drugs)}")

    # 3. Create a clean mapping dictionary from Drug ID -> Contiguous Integer Index (0 to N-1)
    drug_to_idx = {drug: idx for idx, drug in enumerate(all_drugs)}
    idx_to_drug = {idx: drug for drug, idx in drug_to_idx.items()}

    # 4. Map d1 and d2 to integer indices for the GNN edge_index
    df["d1_idx"] = df["d1"].map(drug_to_idx)
    df["d2_idx"] = df["d2"].map(drug_to_idx)

    # 5. Integrity Check: Verify zero missing values or out-of-bounds indices
    assert df["d1_idx"].isnull().sum() == 0, "Error: Found unmapped node indices in d1!"
    assert df["d2_idx"].isnull().sum() == 0, "Error: Found unmapped node indices in d2!"
    print("Integrity check passed: All edges map successfully to valid node indices.")

    # 6. Save the final clean node mapping and edge list
    node_mapping_path = script_dir / "gnn_node_mapping.csv"
    final_edges_path = script_dir / "gnn_edge_list_ready.csv"

    pd.DataFrame(list(drug_to_idx.items()), columns=["drug_id", "node_index"])\
      .to_csv(node_mapping_path, index=False)
      
    df.to_csv(final_edges_path, index=False)

    print("\n" + "=" * 50)
    print(f"Node mapping saved to: {node_mapping_path}")
    print(f"GNN-ready edge list saved to: {final_edges_path}")

if __name__ == "__main__":
    finalize_gnn_dataset()