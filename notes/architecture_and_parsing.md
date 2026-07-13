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

vi. Drug d
  │
  ├─► Parent SMILES ──► SMILES Transformer ──► parent_chem_vec (always computed)
  ├─► Parent Graph  ──► GCN                ──► parent_graph_vec (always computed)
  │        [concat/fuse → parent_vec, dim D]
  │
  ├─► has_metabolite = HMDB_lookup(d)   # binary flag, precomputed offline
  │
  ├─ if has_metabolite == 1:
  │     Metabolite SMILES ──► SMILES Transformer (same weights as parent) ──► metab_chem_vec
  │     Metabolite Graph  ──► GCN (same weights)                          ──► metab_graph_vec
  │     metab_vec = fuse(metab_chem_vec, metab_graph_vec)     # dim D
  │
  ├─ else:
  │     metab_vec = zero_vector(dim D)
  │
  └─► gate = has_metabolite   # literally the same scalar, reused as the mask



vii. Where exactly the gate acts — this is the one design choice to get right. Two options:

(a) Additive/concatenation gating: drug_repr = parent_vec + gate * metab_contribution, where metab_contribution comes from a small learned projection of metab_vec. When gate=0, this is mathematically identical to a parent-only model for that drug. Clean, easy to ablate, easy to explain.- simple and more interpretable. 

(b) Gated cross-attention: metabolite vector participates as an extra key/value in the cross-attention fusion between Drug A and Drug B, but the attention weights into it are forced to zero (or the metab K/V rows are masked out) when gate=0.- ASK THIS 

VIII. eight sharing is mandatory, not optional. Use the same SMILES transformer and GCN for parent and metabolite encoding- HALF OF PARAMTERS- TWO INPUT STREAM 

IX.At the pair level, cross-attention runs between drug_repr_A and drug_repr_B (each already gated per-drug) exactly as in a standard two-stream setup — you don't need Jaanya's four separate cross-attention computations. This is the main complexity reduction versus her original.
Log the gate state per pair (both_have_metabolite, one_has, neither_has) at data-loading time — this is literally your evaluation split, so build it into the dataloader from day one rather than reconstructing it at eval time.

**constraints:**
HMDB structures may not match the metabolite that's actually pharmacologically active. HMDB is curated from a metabolomics angle — you may get a metabolite, not necessarily the primary hepatic one Jaanya was targeting (like NAPQI specifically). Spend part of week 1 manually checking 10-15 well-known drugs (paracetamol, warfarin, etc.) against what HMDB actually returns before committing to it as your source. If it's noisy, consider supplementing with a small hand-curated list of major CYP450 metabolites for known-dangerous drugs, at least for your qualitative case-study section.

- if not using hmdb- propose biotransformer -


 MetaCyc — pathway-oriented small-molecule databases, free access, but coverage skews toward endogenous/well-studied metabolic pathways rather than xenobiotic (drug) metabolism specifically. Worth a quick coverage check against your 645-drug list before committing.




metabolite in cross attention - 
choice b 
This does give you the specific "metabolite atom attends to parent atom" interpretability that made Jaanya's pitch compelling (and is closer to your original "metabolite-aware cross-attention fusion" framing) — at the cost of variable-shaped computation per pair depending on which gates are open, which is more engineering work (masking, padding, or conditional branching in the batch).

choice a 
Here, metabolite info is baked into each drug's single vector before pairing. There is no separate metabolite-to-metabolite attention map — it's implicitly mixed in. Simple, cheap, but you lose the ability to specifically say "the model attended atom X in Metabolite A to atom Y in Parent B."




 build the architecture so it inherently supports the 3-way split (both/one/neither) since that costs you nothing extra — the per-drug gate already gives you this for free. Then treat "which buckets we report and emphasize in the final pitch" as a purely analysis-time decision you can make later, post-mentor-input, without touching the model. That way you're not blocked, and you're not pre-committing to an evaluation framing you might have to walk back.

metabolites in cross atenntion

design a 
 drug_repr_A is not the parent alone — when gate_A=1, it's a fused vector containing metabolite information, and that fused vector is what goes into cross-attention. So cross-attention is seeing metabolite-influenced representations, and Q/K/V inside that attention are computed from vectors that have metabolite signal baked in. The interaction it detects between A and B can absolutely be driven by metabolite features that got folded into drug_repr_A.
What design (a) does not give you is atom-level attribution specifically to the metabolite — you can't point at an attention weight and say "this is Metabolite A's atom talking to Parent B's atom" because that atom-level identity gets blurred once you sum/fuse metab_vec into drug_repr before pairing. You get "the drug-pair interaction was likely metabolite-influenced" at the drug level, not "this specific metabolite atom is responsible" at the atom level. That's the actual tradeoff — not "no metabolite info reaches attention," but "coarser interpretability granularity." I should've been precise about that the first time.





Design (b) is the "keep metabolite identity separate all the way through" version — closer to what Jaanya originally proposed, but with your gate deciding which of the four attention pathways actually get computed for a given pair.
Architecture
Instead of fusing metabolite info into a single per-drug vector before pairing, you keep four separate representations per pair (when available) and run cross-attention between specific combinations of them:
Per drug d:
  parent_vec_d  = fuse(SMILES_transformer(parent_SMILES), GCN(parent_graph))   # always exists
  metab_vec_d   = fuse(SMILES_transformer(metab_SMILES), GCN(metab_graph))     # exists only if gate_d = 1
  gate_d        = has_metabolite(d)   # 0/1, from BioTransformer or your chosen source

For a pair (A, B):
  attn_PP = cross_attention(parent_vec_A, parent_vec_B)                          # always computed
  attn_MP = cross_attention(metab_vec_A,  parent_vec_B)   if gate_A == 1         # metabolite A ↔ parent B
  attn_PM = cross_attention(parent_vec_A, metab_vec_B)    if gate_B == 1         # parent A ↔ metabolite B
  attn_MM = cross_attention(metab_vec_A,  metab_vec_B)    if gate_A==1 and gate_B==1  # metab ↔ metab

  available_outputs = [attn_PP] + [attn_MP if gate_A] + [attn_PM if gate_B] + [attn_MM if both]
  fused = pool_or_weighted_sum(available_outputs)   # e.g. mean over whatever's available, or a small learned gate-weighted combination
  prediction = MLP(fused)
The key structural difference from design (a): attention weights are computed between molecule-specific vectors, not pre-fused drug-level vectors. So attn_MP genuinely tells you which atoms in Metabolite A are attending to which atoms in Parent B — you retain atom-level, entity-specific attribution.
Handling the variable number of active pathways
This is the actual engineering cost of (b). Depending on the pair, you have 1, 2, 3, or 4 active cross-attention outputs (never 0 — attn_PP always runs). You need a pooling/fusion step that works regardless of how many are present:

Simplest: masked mean. Average whatever attention outputs exist, ignoring absent ones. Cheap, no extra parameters, but treats all present pathways as equally important.
Better: gated weighted sum. A small learned scalar weight per pathway type (w_PP, w_MP, w_PM, w_MM), applied only to active pathways and renormalized. Lets the model learn e.g. "MM pathway, when present, tends to matter more for toxicity-type interactions" — closer to Aarya's "contribution analysis" idea, and gives you another interpretability artifact (which pathway did the model lean on).
Either way, batching is the annoying part: within a single batch you'll have pairs with different numbers of active pathways, so you can't just stack tensors naively — you need to compute all four pathways with masking (compute attn_MP/PM/MM using zero-vectors for absent metabolites, then multiply the output by the gate before pooling) rather than skip them conditionally per-example. This is more GPU-forgiving than literally branching per example, but means you're computing some wasted zero-vector attention passes — fine at your data scale, just worth knowing why.





How you actually prove metabolites help — the ablation, concretely
This is the part that answers "will we compute cross-attention for parent-only separately and compare" — yes, exactly that, and here's how to do it without training two separate models:
Train one model, with the gate live during training (gate = actual has_metabolite flag, so the model learns to use metabolite info when available). Then at evaluation time, run the same trained model twice on the metabolite-complete pairs:
Pass 1 (real):        gate_A = 1, gate_B = 1  →  drug_repr uses real metabolite  →  AUROC_with_metab
Pass 2 (counterfactual): gate_A = 0, gate_B = 0  →  drug_repr = parent_vec only    →  AUROC_without_metab
Same weights, same trained model, only the gate is forced off in pass 2. This isolates the causal contribution of metabolite information, because everything else (learned weights, biological branch, fusion) is identical between the two passes. If AUROC_with_metab > AUROC_without_metab on the same pairs, that's a clean, defensible result — you're not comparing two differently-trained models or two different subsets, you're comparing the same model with a feature switched on vs off on the same test examples.
This is a stronger design than what I implied before (comparing metabolite-complete vs. metabolite-incomplete pairs, which conflates "this pair had metabolite data" with "other pairs are just different pairs"). The forced-gate-off counterfactual is the version that actually isolates metabolite contribution as a variable, holding the pair and the model fixed. I'd lead with this in your results section — it's the real "does metabolite-awareness help" experiment, and the metabolite-complete-vs-incomplete comparison becomes a secondary, supporting analysis (useful for showing real-world data-availability tradeoffs, but not your main causal claim).
One thing to decide: should the gate be trainable/stochastic during training (e.g., randomly drop metabolite info for some metabolite-complete pairs during training, like dropout) so the model doesn't become overly dependent on always having it, which would make the forced-off counterfactual unfairly bad? I'd suggest yes — apply gate-dropout (e.g., randomly zero the metabolite pathway 20-30% of the time during training even when data exists) so the parent-only pathway stays well-trained and the counterfactual comparison is fair rather than crippled.



The direct answer: an updated, larger TwoSIDES already exists
The Decagon-standard 645-drug/63,473-pair set you've been using is actually a filtered subset of the original 2012 Tatonetti TWOSIDES data, curated down for statistical reliability (minimum 500-pair support per side-effect type). But the Tatonetti Lab has since updated and expanded this: nSIDES, the current home for their drug side-effect and interaction resources, now hosts a TwoSIDES covering over 3,300 drugs and roughly 63,000 combinations — about 5x more unique drugs while keeping a similar total pair count to what you're used to. It's the same lineage/methodology as the Decagon-era data, just refreshed and broadened, and it's freely downloadable at nsides.io.
This is very likely your answer: swap your parent-drug pool from the narrow 645-drug Decagon cut to this larger ~3,300-drug nSIDES version. You stay within the same well-established dataset family (so you can still cite/compare against Decagon-style baselines credibly), but you get 5x the chemical diversity to intersect against INTEDE (4,701 drugs) and LAGOM (5,322 compounds) — which directly attacks the exact problem we flagged earlier: thin metabolite-complete pair counts.

why use this approach 

If you go back to "only train on metabolite-complete pairs," and separately (or not at all) compare to a parent-only model, you're comparing two different models — possibly different training data volumes, different regularization behavior, different convergence — not the same model with one feature switched on/off. Any AUROC difference you find could be explained by a dozen things other than metabolite information itself.
With the gate, you train one model, then evaluate it twice on the same metabolite-complete pairs — once with the gate live, once forced off. Every other variable (weights, architecture, training data, convergence) is held constant. That isolates metabolite contribution as a variable in a way a separately-trained subset model cannot. This is true no matter how large your metabolite-complete bucket gets — it's a better experiment design, not a scale patch.



intende 
choose the metabolite cloest to parent- choice 

Lagom

Per the paper's own description, expect columns roughly corresponding to: parent SMILES, metabolite SMILES, and likely some identifier/name field. Important curation details already baked in (useful to know so you don't reapply them redundantly): compounds are restricted to specific elements (C, O, N, Cl, F, S, P, Br, I), and parent-metabolite pairs are filtered by Tanimoto similarity > 0.2 (1024-bit Morgan fingerprints) to exclude spurious/unrelated pairings.