1. Drugbank data non accessible; 
here are alternate data sources:
parent drug smiles- pubchem
metabolite structure: best is HMDB- human metabolome: they link back to parent drugs as well- Dont have clean parent ids, have to search and read. 
documented mechanism to cross check in interpretability- PharmGKB- probably actually better than drugbank mechanisms lol. they are diagrams. 

2. Approach without drugbank:
i. around 800 have metabolite structure, 
ii. 150-400 have proper mechanisms with text - can i just stop here? 
iii. 30-60 hand verifyable- use this in the validation part - this is the real picture. 
and a couple hundred to bulk train with . 


metabolite linking approach (possible - just a thought for now (15-20 min extra work) )
1. search hmdb
2. cross check against pharmGKB
3. get strcuture from pubchem 

dataset info:
pubchem- 
smiles tring feed into text transformer and rdkit - aaryas spatial graph nodes 

hmdb 
donwloads section - all metabolites 
write a parsing script using Python's pandas or xml.etree that reads the HMDB file and builds a simple lookup dictionary matching Parent Name/CID → Metabolite SMILES.

pharm gkb 
d types: Pharmacokinetics (PK) and Pharmacodynamics (PD).
can also use to flag aaryas protein 
specific genes and liver enzymes that catalyze the reaction shown.
 The diagram traces whether a metabolite safely goes to renal clearance or transforms into a highly reactive toxic byproduct (like explicitly tracking NAPQI hitting cellular proteins).

 decagon- pre cleaned- but has stitch ids- clean to get only numeric part 
 open source files - pre cleaned - have perfectly documented metabolites, 


 **important notes/ traps**
 1. Different drugs have diff lengths of smiles strings- to avoid wasting gpu memory- 
 explicitly create a src_key_padding_mask inside the PyTorch dataset loader. Pass this mask directly into the cross-attention layer so the query matrices completely ignore empty padding tokens.
 2. molecule can be canonical? isomeric SMILES- Natively enforce Chem.MolToSmiles(mol, isomericSmiles=False) in your data pipelines to guarantee every string across all 4 parallel text/graph branches drops to the identical canonical standard.
 3. metabolite - primary wont be specified like it was in drug bank. 
