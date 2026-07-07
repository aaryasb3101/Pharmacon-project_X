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
