import pandas as pd
from sklearn.model_selection import train_test_split #machine learning library 
#train_test_split splits the data

#giving the input file
INPUT_FILE = "final_metabolites.csv"

#expected output file
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

# Load dataset (pandas function that reads the file and converts it into a dataframe)
df = pd.read_csv(INPUT_FILE)

print("Total records:", len(df)) #len(df) displays the number of rowsin the dataframe

# Remove rows without a parent ID
df = df.dropna(               #dropna removes rows with NaN, so essentially we remove the rows with no parent id
    subset=["parent_drug_id"] #tells the function to only check the parent drug id coloumn for missing values
).reset_index(drop=True) #the rows we dropped creates a gap in index numbers like 1,3,5 so2 and 4 were removed, this fixes that

# Get unique parent drugs
parent_ids = df["parent_drug_id"].unique() #we make an array of unique parent drug ids and store it in parent_ids

print("Unique parent drugs:", len(parent_ids)) #print the number of unique parent drug ids obtained

# Split parent drugs
train_parents, temp_parents = train_test_split(  #train_parents receive the first output given by the function train_test_split and the rest gets stored in temp_parents temporarily
    parent_ids,
    test_size=0.20, #20 percent goes to temp_parents and remaining 80 percent goes into our training parent drug ids
    random_state=42, #controls random split so if the input/order is same it will give us the same set, used for reproducibility
)

val_parents, test_parents = train_test_split(
    temp_parents,
    test_size=0.50,
    random_state=42,
)

# Create datasets
train_df = df[
    df["parent_drug_id"].isin(train_parents) #checks weather each parent drug id is in train_parents and adds it to the training dataset if true
].reset_index(drop=True)

val_df = df[
    df["parent_drug_id"].isin(val_parents)
].reset_index(drop=True)

test_df = df[
    df["parent_drug_id"].isin(test_parents)
].reset_index(drop=True)

# Save
train_df.to_csv( #writing the dataframe to a csv file
    TRAIN_FILE,
    index=False, #don't write the pandas row index into the csv
)

val_df.to_csv(
    VAL_FILE,
    index=False,
)

test_df.to_csv(
    TEST_FILE,
    index=False,
)

# Summary
print("\nDataset split:")
print(
    f"Train: {len(train_df)} records "
    f"| {len(train_parents)} parents"
)

print(
    f"Validation: {len(val_df)} records "
    f"| {len(val_parents)} parents"
)

print(
    f"Test: {len(test_df)} records "
    f"| {len(test_parents)} parents"
)

# Verify no parent overlap
#we make sets for training, validation and testing
train_set = set(train_parents) 
val_set = set(val_parents)
test_set = set(test_parents)

print("\nParent overlap checks:")

print(
    "Train ∩ Validation:",
    len(train_set & val_set)
)

print(
    "Train ∩ Test:",
    len(train_set & test_set)
)

print(
    "Validation ∩ Test:",
    len(val_set & test_set)
)