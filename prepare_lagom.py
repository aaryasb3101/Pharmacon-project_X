import pandas as pd
from pathlib import Path

def prepare_smiles():
    script_dir = Path(__file__).resolve().parent
    
    # Try looking for your final node mapping or raw drugs file
    mapping_path = script_dir / "gnn_node_mapping.csv"
    drugs_path = script_dir / "drugs.csv"
    
    if drugs_path.exists():
        df = pd.read_csv(drugs_path)
        smiles_col = "smiles" if "smiles" in df.columns else "SMILES"
        unique_smiles = df[smiles_col].dropna().unique()
    elif mapping_path.exists():
        # Fallback if extracting from node mapping
        df = pd.read_csv(mapping_path)
        unique_smiles = df["drug_id"].dropna().unique() # or your smiles column
    else:
        print("Error: Could not find dataset source file.")
        return
        
    output_file = script_dir / "lagom_input_smiles.txt"
    pd.Series(unique_smiles).to_csv(output_file, index=False, header=False)
    print(f"Successfully exported {len(unique_smiles)} unique SMILES strings to: {output_file}")

if __name__ == "__main__":
    prepare_smiles()