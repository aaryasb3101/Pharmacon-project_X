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
They provide clean, open tsv (tab-separated values) down-loaders. You can download their clinical annotation datasets or "Relationships" file directly
d types: Pharmacokinetics (PK) and Pharmacodynamics (PD).
can also use to flag aaryas protein 
specific genes and liver enzymes (like CYP2E1, CYP2C19, or CYP2D6) that catalyze the reaction.The Toxicity & Clearance Destinies: The diagram traces whether a metabolite safely goes to renal clearance or transforms into a highly reactive toxic byproduct (like explicitly tracking NAPQI hitting cellular proteins).