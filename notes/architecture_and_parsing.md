July 6: 
1. Got the academic approval from drugbank to access the website- 
navigated- found that the curated dataset containing each parent drug, metabolite and mechanism information from my side of the approach- **will approximately have 500-800 drugs WITH ALL INFO NEEDED- properly mapped. This means around 200,000 ddi pairs for our dataset- scope.**

2. UNIPROT IDs: looked over how to connect aaryas protein approach to original problem statement since she didnt mention anything specifcally in her proposal. 

- **UNIPROT - universal iD**
- can get acid sequence used in esm 
- can go to reactome to get pathway 
- STRING- does not have direct ID- we translate the uniprot to strings internal naming- and find PPIs.

**Benefit:**
UNIPROT will make it simpler to streamline approach- instead of what aarya said to backtrack and cross check all protein pathways if they are present in DRUGBANK, we instead can just take the proper set of drugs and metabolites and just mapp to those exact proteins and pathways. 

3. **DATA- will download full XML file in drugbank**
 Went to drugbanks official site-  FOUND OUT THEY DISABLED ANY DATA DOWNLOADS TEMPORARILY. 

need to look at alternate data approaches- although the entire metabolite/ protein approach was based on drugbanks extra features- other websites barely have enough parent drug data itself. 

4. **Researched Parser approaches:**
**lxml.etree.iterparse**- seems like the best approach. 
- faster than std lib 
- elem.clear and del elem.get parent- this clears and hands blocks one at a time instead of loading together. 

5. Question: 
since i am not able to access data- to create the parser i dont know if the <metabolite> has SMILES itself, or if just name and ID, and we must find somewhere else in file. 

6. Approach: 
1. Will download full XML- get the subset with metabolite 
2. write gcn and transformer code and then on the first 20-50 rows in csv/json of subset- iterate to make sure the code works. 

NEW APPROACH thoughts-
KEEPING METABOLITES prevents it from being a remake of decagon  
1. a two-stage Pre-training & Fine-tuning pipeline- 30,000 pairs to pre train graph and encoders, then freeze them to have a 4000 pair mechanism to train cross attention - this prevents overfitting 
2. cross attention is applied to all- but when we isolate the 4k pairs- AUROC or AUPRC on these.- Interpretability Evaluation script
3. DATA SIZE: The standard TWOSIDES benchmark features roughly 46,000 total interactions. Operating at 30,000 pairs ensures your project scales alongside published deep learning work.
4. **chemBERTa**- Ai model, understands chemistry- trained on 77 million chemical compounds. so that 30kpairs isnt overfit - 
i. GCN and cross attention can be focused on 
5. need to clean up to remove inorganics, biologics 
6. **how do ik what metabolite to choose:** in my proposal- i said ill choose the primary one- not most abundant: but non drugbank:
1. inHMDB XML-look for the field tracking "Biomarker / Concentration Type" or check the text description. programmatically select the Major Primary Metabolite
2. primary metabolite will have very similar structure to parent as only one transformation step has happened: 
RDKit Tanimoto Similarity- more than 0.6 
3. matching words like "toxic", "active metabolite",- check in description 


 Python matching loop with this exact order of operations:
If the drug has an entry in HMDB, look at the <metabolite_associations> block.
If multiple structures exist, pick the first one that possesses a valid SMILES string and has a molecular weight lower than the parent drug (ensuring it's a breakdown product, not a complex conjugate).
If a drug completely lacks any mapped structural metabolites, drop the pairs associated with that drug from your 30,000 training pool entirely to prevent passing blank arrays to your cross-attention network.


**decagon vs github- for parent drugs**

1. STITCH IDs in decagon so will have to clean th eid to show only numeric part - uses exact same jcsun-00/Twosides repo. 
2. 645 core drugs- translated already - check with author if they didnt drop any columns while filtering for themselves 


