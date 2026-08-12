import os
import xml.etree.ElementTree as ET
import pandas as pd

# 1. AUTOMATICALLY LOCATE CURRENT DIRECTORY
base_path = os.path.dirname(os.path.abspath(__file__))

DRUG_SMILES_PATH = os.path.join(base_path, "drug_smiles.csv")
DDIS_PATH = os.path.join(base_path, "ddis.csv")
HMDB_XML_PATH = os.path.join(base_path, "hmdb_metabolites.xml")
OUTPUT_PATH = os.path.join(base_path, "four_stream_ddi.csv")

print("🔄 Starting 4-Stream Mapping Engine...")

# 2. LOAD DATA FILES
try:
    df_drugs = pd.read_csv(DRUG_SMILES_PATH)
    df_ddi = pd.read_csv(DDIS_PATH)
    print(f"✅ Loaded {len(df_drugs)} parent drugs.")
    print(f"✅ Loaded {len(df_ddi)} clinical DDI interaction rows.")
except Exception as e:
    print(f"❌ Error loading CSV files: {e}")
    exit()

drug_id_col = 'drug_id'
d1_col, d2_col = df_ddi.columns[0], df_ddi.columns[1]

# 3. PARSE THE HMDB XML FILE
print("🔄 Reading and parsing hmdb_metabolites.xml (Processing entries)...")
hmdb_map = {}

try:
    context = ET.iterparse(HMDB_XML_PATH, events=("end",))
    for event, elem in context:
        if elem.tag.endswith('metabolite'):
            metabolite_smiles = None
            linked_drug_ids = []
            
            for child in elem:
                tag_name = child.tag.split('}')[-1]
                
                if tag_name == 'smiles' and child.text:
                    metabolite_smiles = child.text.strip()
                
                # Capture DrugBank IDs and PubChem compound IDs
                elif tag_name in ['drugbank_id', 'pubchem_compound_id'] and child.text:
                    val = child.text.strip().upper()
                    linked_drug_ids.append(val)
                    
                    # If it's a numeric PubChem ID, strip leading zeros to normalize it
                    if val.isdigit():
                        linked_drug_ids.append(str(int(val)))
            
            if metabolite_smiles and linked_drug_ids:
                for d_id in linked_drug_ids:
                    hmdb_map[d_id] = metabolite_smiles
            
            elem.clear() # Free memory
            
    print(f"✅ Extracted {len(hmdb_map)} database keys from HMDB.")
except Exception as e:
    print(f"❌ Error parsing HMDB XML: {e}")
    exit()

# 4. MAP INDIVIDUAL DRUGS USING NORMALIZED NUMERICS
print("🔄 Intersecting your CID drugs with HMDB structures...")
valid_drugs = set()

for _, row in df_drugs.iterrows():
    raw_id = str(row[drug_id_col]).strip()
    
    # Strip away 'CID', 'CID0', and leading zeros to get the pure numeric string
    # E.g., 'CID000002173' becomes '2173'
    clean_numeric = ''.join(filter(str.isdigit, raw_id))
    if clean_numeric:
        normalized_id = str(int(clean_numeric))
        
        # Check if this pure number exists inside HMDB's cross-references
        if normalized_id in hmdb_map:
            valid_drugs.add(raw_id)

print(f"🎯 Individual Mapping Results: {len(valid_drugs)} out of {len(df_drugs)} drugs successfully matched to an HMDB metabolite.")

# 5. FILTER THE DDI PAIRS (THE 4-STREAM CONSTRAINT)
print("🔄 Filtering DDI matrix for valid 4-stream pairs...")
df_filtered = df_ddi[
    df_ddi[d1_col].astype(str).str.strip().isin(valid_drugs) & 
    df_ddi[d2_col].astype(str).str.strip().isin(valid_drugs)
]

unique_pairs = df_filtered[[d1_col, d2_col]].drop_duplicates().shape[0]

# 6. OUTPUT METRICS SUMMARY
print("\n" + "="*50)
print("📊 FINAL 4-STREAM QUANTIFICATION REPORT")
print("="*50)
print(f"• Unique Parent Drugs Successfully Mapped: {len(valid_drugs)}")
print(f"• Total Valid Unique Interaction Pairs:    {unique_pairs}")
print(f"• Total Training Rows (Multi-Typed DDI):   {len(df_filtered)}")
print("="*50 + "\n")

# Save file
df_filtered.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Cleaned 4-stream file saved directly to folder:\n   {OUTPUT_PATH}")