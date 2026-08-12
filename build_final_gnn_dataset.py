import pandas as pd
import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    
    # 1. Load your core files
    print("Loading data files...", file=sys.stderr)
    ddi_path = script_dir / "ddis.csv"
    mapping_path = script_dir / "twosides_drugbank_mapping_ultimate.csv"
    
    if not ddi_path.exists() or not mapping_path.exists():
        print("Error: Missing ddis.csv or twosides_drugbank_mapping_ultimate.csv", file=sys.stderr)
        sys.exit(1)
        
    ddi_df = pd.read_csv(ddi_path)
    mapping_df = pd.read_csv(mapping_path)
    
    # 2. Create lookup dictionaries from the mapping file
    # Maps TWOSIDES drug ID -> DrugBank ID and DrugBank Name
    db_id_map = dict(zip(mapping_df["twosides_drug_id"], mapping_df["matched_drugbank_id"]))
    db_name_map = dict(zip(mapping_df["twosides_drug_id"], mapping_df["matched_drugbank_name"]))
    
    # 3. Map Drug 1 (d1) and Drug 2 (d2) in your DDI interaction table
    print("Mapping drug attributes onto DDI interaction pairs...", file=sys.stderr)
    ddi_df["d1_drugbank_id"] = ddi_df["d1"].map(db_id_map)
    ddi_df["d2_drugbank_id"] = ddi_df["d2"].map(db_id_map)
    
    ddi_df["d1_drugbank_name"] = ddi_df["d1"].map(db_name_map)
    ddi_df["d2_drugbank_name"] = ddi_df["d2"].map(db_name_map)
    
    # 4. Filter out any rows where mapping failed completely if you want strict alignment
    # (Optional: keep them if your GNN handles unmapped nodes via raw SMILES)
    enriched_path = script_dir / "ddis_final_enriched_dataset.csv"
    ddi_df.to_csv(enriched_path, index=False)
    
    print("=" * 60)
    print(f"Total DDI interaction pairs processed: {len(ddi_df)}")
    print(f"Final enriched dataset successfully saved to: {enriched_path}")

if __name__ == "__main__":
    main()