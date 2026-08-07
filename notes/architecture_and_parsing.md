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
3. DATA SIZE: The standard TWOSIDES benchmark features roughly 46,000 total interactions. Operating at 30,000 pairs ensures our project scales alongside published deep learning work.
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

-----

approaches as for now: 
1. 
we map the parent drug to a metabolite only if the TEXT description matches in hmdb- longer process and will reduce usable metabolites further.
2. 
this is the most doable and ig most impressive option-
we change our pitch from “we consider both parent and metabolites”- to “ metabolite existence increases our score and proves more accurate than just parent drug models” 
essentially we have take parent drugs as input- have a binary check- does metabolite exist or not. if exists we use it further, if not we just use parent drugs and it falls back by inputting zero vector for the “has metabolite” variable. this way we train on ALL pairs- our dataset stays large and we ahow something different from what decagon/ other papers have done. something i realised was- with this we actually have a baseline that proves that metabolites increase auroc score. we report both aurocs separately- with and without metabolites. 
3. 
using biotransformer- it predicts the metabolite structure- im just listing it here but its not the optimal approach
4. 
we drop metabolites entirely.


------

**WORK ON OPTION 2 practically:**
NOTES:
i. entire approach assumed drugbank access
ii. Reporting AUROC split by metabolite-complete vs. incomplete is probabl a better validation strategy than what either original proposal had
iii. You've turned "we hope metabolites help" into a falsifiable claim. That's a stronger pitch to mentors than either original.- biweekly
iv. hmdb coverage is low- per type breakdown vs: multi label auroc- we should set a minimum-support threshold below which we don't report a split.
v. A drug having an HMDB entry doesn't mean the documented TWOSIDES interaction for that pair is happening via the metabolite. So a higher AUROC on metabolite-complete pairs would be suggestive, not proof of mechanism - no validation possible anymore. 



vii. Where exactly the gate acts — this is the one design choice to get right. Two options:

(a) Additive/concatenation gating: drug_repr = parent_vec + gate * metab_contribution, where metab_contribution comes from a small learned projection of metab_vec. When gate=0, this is mathematically identical to a parent-only model for that drug. Clean, easy to ablate, easy to explain.- simple and more interpretable. 

(b) Gated cross-attention: metabolite vector participates as an extra key/value in the cross-attention fusion between Drug A and Drug B, but the attention weights into it are forced to zero (or the metab K/V rows are masked out) when gate=0.- ASK THIS 

VIII. eight sharing is mandatory, not optional. Use the same SMILES transformer and GCN for parent and metabolite encoding- HALF OF PARAMTERS- TWO INPUT STREAM 


**constraints:**
HMDB structures may not match the metabolite that's actually pharmacologically active. HMDB is curated from a metabolomics angle — you may get a metabolite, not necessarily the primary hepatic one Jaanya was targeting (like NAPQI specifically). Spend part of week 1 manually checking 10-15 well-known drugs (paracetamol, warfarin, etc.) against what HMDB actually returns before committing to it as your source. If it's noisy, consider supplementing with a small hand-curated list of major CYP450 metabolites for known-dangerous drugs, at least for your qualitative case-study section.

- if not using hmdb- propose biotransformer -



metabolite in cross attention - 
choice b 

choice a 
Here, metabolite info is taken into each drug's single vector before pairing. There is no separate metabolite-to-metabolite attention map — it's implicitly mixed in. Simple, cheap, but you lose the ability to specifically say "the model attended atom X in Metabolite A to atom Y in Parent B."



metabolites in cross atenntion

design a 
 drug_repr_A is not the parent alone - when gate_A=1, it's a fused vector containing metabolite information, and that fused vector is what goes into cross-attention. So cross-attention is seeing metabolite-influenced representations, and Q/K/V inside that attention are computed from vectors that have metabolite signal baked in. The interaction it detects between A and B can absolutely be driven by metabolite features that got folded into drug_repr_A.


Design (b) is the "keep metabolite identity separate all the way through" version 
Architecture
Instead of fusing metabolite info into a single per-drug vector before pairing, you keep four separate representations per pair (when available) and run cross-attention between specific combinations of them:



Simplest: masked mean. Average whatever attention outputs exist, ignoring absent ones. 
Better: gated weighted sum. A small learned scalar weight per pathway type (w_PP, w_MP, w_PM, w_MM), applied only to active pathways and renormalized. Lets the model learn e.g. "MM pathway, when present, tends to matter more for toxicity-type interactions"





how to prove metabolites help — the ablation

The Decagon-standard 645-drug/63,473-pair set - actually a filtered subset of the original 2012 Tatonetti TWOSIDES data, curated down for statistical reliability (minimum 500-pair support per side-effect type). But the Tatonetti Lab has since updated and expanded this: nSIDES, the current home for their drug side-effect and interaction resources, now hosts a TwoSIDES covering over 3,300 drugs and roughly 63,000 combinations

why use this approach 

If we go back to "only train on metabolite-complete pairs," and separately (or not at all) compare to a parent-only model, we're comparing two different models - possibly different training data volumes, different regularization behavior, different convergence - not the same model with one feature switched on/off. Any AUROC difference you find could be explained by a dozen things other than metabolite information itself.
intende 
choose the metabolite cloest to parent- choice 

Lagom

Per the paper's own description, expect columns roughly corresponding to: parent SMILES, metabolite SMILES, and likely some identifier/name field. Important curation details already baked in (useful to know so you don't reapply them redundantly): compounds are restricted to specific elements (C, O, N, Cl, F, S, P, Br, I), and parent-metabolite pairs are filtered by Tanimoto similarity > 0.2 (1024-bit Morgan fingerprints) to exclude spurious/unrelated pairings.



transfomrer notes 
custom transformer or should we 
matrices for diff features 
key valu ematrix- weights, vocab, get matrices from pretrained models from molformer- prev knowledge of chemistry - match the embeddingd  to the model we are using - 

take smiles string- tokeniser- sep the individual atoms- takes apart c and cl - embeddings - a vector - 256 dimensions random 256 numbers for one molecule- model- trained- go through it and have 4 layerrs- multihead self attention- normalisation and feedforfroawrd- layers traianed- adjust the embeddings to store info about it- 4=cl- random initialisation changed to 2- multihead self attention- one head one feature- aromatic rings etc- cls- embeddings pull together to a parent or metabolite embedding- molecular embedding- 256 vector- so that same 


1. dataset finalise 
2. what problems during finalisation 
3. 3 files for dataset 
4. transformer actual implememntion: use of it in layman terms 
5. architecture diagram at end 
6. comparion with existing and why ours is diff- molformer touch 
7. future plan- gnn and relation to bio branch 
8. bio branch - ppt bifurcation 
