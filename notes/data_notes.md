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



REPOS FOR DATA :
jcsun-00/Twosides- USING THIS 
yueyu1030/SumGNN

used jcsun- 
neg samples is a column in ddi file from it, 
column contains a randomly generated control drug ID that is mathematically proven not to cause that specific side effect type when combined with. allowing us to train a highly balanced cross-entropy loss function without having to manually generate random contrastive drug pairs ourselves."

1. chembert a
2. biotransformer- predicts metabolite stru for all 
3. include both meta and non meta parents - put up a binary 


why DRUGBANK WOULD HAVE BEEN BETTER 
 DrugBank's own drug pages contain a structured "Metabolites" section explicitly listing each drug's known metabolite structures (with SMILES/InChIKey) — that parent→metabolite edge is exactly what HMDB is missing in its free bulk download, because HMDB originally licensed and displayed that data from DrugBank rather than curating it independently.


intende 
choose the metabolite cloest to parent- choice 

Lagom

Per the paper's own description, expect columns roughly corresponding to: parent SMILES, metabolite SMILES, and likely some identifier/name field. Important curation details already baked in (useful to know so you don't reapply them redundantly): compounds are restricted to specific elements (C, O, N, Cl, F, S, P, Br, I), and parent-metabolite pairs are filtered by Tanimoto similarity > 0.2 (1024-bit Morgan fingerprints) to exclude spurious/unrelated pairings.


keep in mind 
don't wait for InChIKey matching to be perfect before you get your coverage number. Run the full pipeline once, even sloppily, get a number for % of 645 drugs matched, and only invest more time in the fuzzy-matching/spot-check refinement if that number is borderline 



---- NEW NOTES FROM DRUGBANK XML- AVANISH----

=== Full structure of first <drug> element (depth-limited to 4) ===

<drug>
  <drugbank-id>  = "DB00001"
  <name>  = "Lepirudin"
  <description>  = "Lepirudin is a recombinant hirudin forme..."
  <cas-number>  = "138068-37-8"
  <unii>  = "Y43GF64R34"
  <state>  = "solid"
  <groups>
    <group>  = "approved"
  <general-references>
    <articles>
      <article>
        <ref-id>  = "A1"
        <pubmed-id>  = "16244762"
        <citation>  = "Smythe MA, Stephens JL, Koerber JM, Matt..."
    <textbooks>
    <links>
      <link>
        <ref-id>  = "L48"
        <title>  = "Google books"
        <url>  = "http://books.google.com/books?id=iadLoXo..."
    <attachments>
  <synthesis-reference>  = "Recombinant hirudin expressed by using y..."
  <indication>  = "Lepirudin is indicated for anticoagulati..."
  <pharmacodynamics>  = "Lepirudin is a recombinant hirudin that ..."
  <mechanism-of-action>  = "Lepirudin is a direct thrombin inhibitor..."
  <toxicity>  = "The acute toxicity of intravenous lepiru..."
  <metabolism>  = "As a polypeptide, lepirudin is expected ..."
  <absorption>  = "Lepirudin administered as a single intra..."
  <half-life>  = "Lepirudin has an initial half-life of ap..."
  <protein-binding>  = "In human plasma, the protein binding of ..."
  <route-of-elimination>  = "Lepirudin is mostly excreted through uri..."
  <volume-of-distribution>  = "The volume of distribution of lepirudin ..."
  <clearance>  = "The clearance of lepirudin is proportion..."
  <classification>
    <description>
    <direct-parent>  = "Peptides"
    <kingdom>  = "Organic Compounds"
    <superclass>  = "Organic Acids"
    <class>  = "Carboxylic Acids and Derivatives"
    <subclass>  = "Amino Acids, Peptides, and Analogues"
  <salts>
  <synonyms>
    <synonym>  = "[Leu1, Thr2]-63-desulfohirudin"
  <products>
    <product>
      <name>  = "Refludan"
      <labeller>  = "Bayer Ag"
      <ndc-id>
      <ndc-product-code>  = "50419-150"
      <dpd-id>
      <ema-product-code>
      <ema-ma-number>
      <started-marketing-on>  = "1998-03-06"
      <ended-marketing-on>  = "2013-06-30"
      <dosage-form>  = "Powder"
      <strength>  = "50 mg/1mL"
      <route>  = "Intravenous"
      <fda-application-number>  = "NDA020807"
      <generic>  = "false"
      <over-the-counter>  = "false"
      <approved>  = "true"
      <country>  = "US"
      <source>  = "FDA NDC"
  <international-brands>
  <mixtures>
  <packagers>
    <packager>
      <name>  = "Bayer Healthcare"
      <url>  = "http://www.bayerhealthcare.com"
  <manufacturers>
    <manufacturer>  = "Bayer healthcare pharmaceuticals inc"
  <prices>
    <price>
      <description>  = "Refludan 50 mg vial"
      <cost>  = "273.19"
      <unit>  = "vial"
  <categories>
    <category>
      <category>  = "Amino Acids, Peptides, and Proteins"
      <mesh-id>  = "D000602"
  <affected-organisms>
    <affected-organism>  = "Humans and other mammals"
  <dosages>
    <dosage>
      <form>  = "Injection, powder, for suspension"
      <route>  = "Intravenous"
      <strength>  = "5000000 mg"
  <atc-codes>
    <atc-code>
      <level>  = "Direct thrombin inhibitors"
  <ahfs-codes>
  <pdb-entries>
  <patents>
    <patent>
      <number>  = "5180668"
      <country>  = "United States"
      <approved>  = "1993-01-19"
      <expires>  = "2010-01-19"
      <pediatric-extension>  = "false"
  <food-interactions>
    <food-interaction>  = "Avoid herbs and supplements with anticoa..."
  <drug-interactions>
    <drug-interaction>
      <drugbank-id>  = "DB06605"
      <name>  = "Apixaban"
      <description>  = "Apixaban may increase the anticoagulant ..."
  <sequences>
    <sequence>  = ">DB00001 sequence
LTYTDCTESGQNLCLCEGSNVC..."
  <experimental-properties>
    <property>
      <kind>  = "Water Solubility"
      <value>  = "Soluble"
      <source>  = "Health Canada drug label"
  <external-identifiers>
    <external-identifier>
      <resource>  = "Drugs Product Database (DPD)"
      <identifier>  = "11916"
  <external-links>
    <external-link>
      <resource>  = "RxList"
      <url>  = "http://www.rxlist.com/cgi/generic/lepiru..."
  <pathways>
    <pathway>
      <smpdb-id>  = "SMP0000278"
      <name>  = "Lepirudin Action Pathway"
      <category>  = "drug_action"
      <drugs>
        <drug>
      <enzymes>
        <uniprot-id>  = "P00734"
  <reactions>
    <reaction>
      <sequence>  = "1"
      <left-element>
        <drugbank-id>  = "DB00001"
        <name>  = "Lepirudin"
      <right-element>
        <drugbank-id>  = "DBMET03462"
        <name>  = "M1 (1-64)"
      <enzymes>
  <snp-effects>
  <snp-adverse-drug-reactions>
  <targets>
    <target>
      <id>  = "BE0000048"
      <name>  = "Prothrombin"
      <organism>  = "Humans"
      <actions>
        <action>  = "inhibitor"
      <references>
        <articles>
        <textbooks>
        <links>
        <attachments>
      <known-action>  = "yes"
      <polypeptide>
        <name>  = "Prothrombin"
        <general-function>  = "Thrombin, which cleaves bonds after Arg ..."
        <specific-function>  = "calcium ion binding"
        <gene-name>  = "F2"
        <locus>  = "11p11.2"
        <cellular-location>  = "Secreted, extracellular space"
        <transmembrane-regions>
        <signal-regions>  = "1-24"
        <theoretical-pi>  = "5.7"
        <molecular-weight>  = "70036.295"
        <chromosome-location>  = "11"
        <organism>  = "Humans"
        <external-identifiers>
        <synonyms>
        <amino-acid-sequence>  = ">lcl|BSEQ0016004|Prothrombin
MAHVRGLQLPG..."
        <gene-sequence>  = ">lcl|BSEQ0016005|Prothrombin (F2)
ATGGCG..."
        <pfams>
        <go-classifiers>
  <enzymes>
  <carriers>
  <transporters>


  - HERE METABOLITE INFO IS IN THE FORM OF A REACTION- LEFT IS PARENT, RIGHT METABOLITE 

  2. PARSED TOTAL SITE: 
  Total drugs scanned:                     17430
  'Proper' organic drugs (has SMILES):     12313
    ...of those, with >=1 metabolite rxn:  935
      ...percentage with metabolites:        7.6%

      3. CHECKED IF METABOLITE DATA EXISTS ANYWHERE ELSE- NO WHERE. RAN FOUR SCRIPTS. 
      4. drugbank orginallly maybe only also contained ids only and not strcutures

      --WHAT THIS MEANS FOR PROJECT:---
      1. metabolite structures not there- only name and id, we still need to look up structures from other webiste- can check hmbd to see if ids get mapped properly. (smiles/inchikey not present so there is no actual data we can work on w this). this may reduce metabolite data more
      can use drugbank to make flag- metabolite exists or not. 
      2. parent drug scale goes from 645 to 12k- biologic like drugs if removed and these 12k have smiles
      3. check how useful metabolite id data is to us 
      4. lagom is actually not a lookup database- its a transformer based predictive model which actually trained on drugbank data itself to create its database (db+ MetXBioDB)
      5. twosides- 943 predictive labels for side effetcts: we will have to create names/labels for each side effect (free-text mechanism sentences exist in drugbank)- manual work need to discuss if we need or not 


      how to check metabolite ids: 
      1. Direct ID cross-reference (best case, if it exists)
     2.  Name-based matching-  DrugBank <reactions> gives you a metabolite name
     3. Structural cross-check once we have a candidate match


HMDB entries do carry a direct
DrugBank ID field.

1. lagom org training dataset unavaibale as no drugbank access 
2. try met x
or use lagom predictive 
or check biancas preprocesss set 
3. mapping twosides drugs to drugbank 
4. 

mapping: 
DrugBank scan complete: 17430 total drugs, 12313 organic (SMILES) drugs

1. found that hmdb+ drugbank not mapping: hmdbs id means it is the exact drugbank drug not the metabolite - hence cant use hmdb 
2. twosides- only two or do i also include db- that means we will predict not just properly mapped labels 
3. lagom: the repo i sent vs what we can use now: have to map parent drug 
4. metbiox thingy check 
5. ran similarity on 3k drugs- high similarity 3080- "1-Methylhistidine € metabolite of 1-Methylhistidine
* (DB04151)" - the parent and the
* "metabolite" are the same exact compound


1. parent drugs do we want to just keep the 645 drug set or 12k (from drugbank) - the 12k drugs wont have labels mapped to all of them so basically our model will become a side effect predictive model- whereas if we just use 645 - we have exact labels for each. again, decagon uses only 645 so lmk what we should do 
2. transformer- do we train it now of kaggle or gpus are avaiable? 
3. will start looking at the biological data- even if we dont include in next biweekly 
4. are separate miniprojects expected for this biweekly?
ok so for gnn- i went over the resource you sent - talks about it conceptually and visually well- do you have any recs for something that shows  1. molecular graph guidance like decagon,2. something that shows batching variable sized graphs/ stacking depth/multi relational gnn

26 july meet:

molformer 
Nah separate mini projects are not expected now, but you r working on the datasets and aarya is working on the transformer
look into pretrained models as last resort- 
And yes look into how you'll be implementing biological branch and explainability thoda so that no conflicts arise in the future, next meet me mention bio branch, let's keep explainability for a later meet
Metabolites ke case me even if you use simulated methods like lagom, it's good enough to prove a hypothesis
It's not a bad outcome, but it's upto you depending on what you want to do here ig
But it will become significantly harder for you
645 drugs ka corpus is tried and true in decagon- use twosides 
Let's go with TWOSIDES for now
GNN graph banane me bhi easy (relatively) padega 
Y'all will have to deal with bio branch also later
biologically ig we may need a gate too - because we wont have all pathways

29 july: 
topics:

1. The lagom repo was the dataset that lagom trained on but it references DB whcih we dont have- and thus we had to resort to using the model- thus the metabolite structures are predicted and not solid data. - can check metbiox but less likelihood 
 2. Parent drug dataset - questions that arose and what i found out 
a. Do i use Twosides 645 only or include all 12k from Drugbank? Twosides better- already cleaned data- better baseline as decagon uses it. If we include all 12k then our model becomes prediction based as all arent mapped in drugbank 
3. 
3. HMDB? - cannot use for metabolites : it was giving very high accuracie as the same molcule was mapped as metabolite and parent drug- 
1-Methylhistidine  DB04151. HMDB's drugbank id field might just mean "this exact same chemical compound also happens to exist in DrugBank's own catalog" , not "this is a metabolite derived from that DrugBank parent drug." a match doesn't automatically mean an actual parent-drug-to-metabolite relationship.



What each file contains

ddis_final_enriched_dataset.csv
d1, d2, type, Neg samples, d1_drugbank_id, d2_drugbank_id, d1_drugbank_name, d2_drugbank_name

Interaction labels present — type is your label column
This is your TWOSIDES pairs, enriched with the DrugBank mapping you built earlier (this confirms your TWOSIDES→DrugBank join worked
No SMILES here — that's fine, this file's job is interaction labels + ID cross-referencing, not structure

gnn_edge_list_ready.csv
Same columns as above, plus d1_idx, d2_idx

This adds the integer node indices needed for GNN libraries (PyTorch Geometric expects edges as integer index pairs, not string IDs)
Still no SMILES — also expected, since an edge list only describes connections between nodes, not node features

gnn_node_mapping.csv
drug_id, node_index

Just the lookup table: which integer index corresponds to which drug — necessary for converting between your GNN's internal indexing and real drug identities

This is the key thing to clarify first: is gnn_edge_list_ready.csv representing:
(a) the DDI interaction graph — nodes = drugs, edges = "these two drugs interact" (which is what it looks like, given d1_idx, d2_idx per interaction pair), or
(b) molecular graphs per drug — nodes = atoms, edges = bonds, one graph per drug

Based on what you've shown me, this is clearly (a) — a drug-drug interaction graph, not a per-molecule atomic graph. This matters a lot for your answer.

Why this changes things

If your architecture's GCN was meant to encode molecular structure (atoms/bonds within a single drug, as you described earlier — "SMILES Transformer + GCN → parent_chem_vec"), then this gnn_edge_list_ready.csv file is a different graph entirely — the DDI relationship graph, not the per-molecule atomic graph your chemical branch needs. These serve two separate architectural purposes and shouldn't be confused with each other.

What you actually need for the SMILES Transformer (and per-molecule GCN)

A simple file: drug_id (or node_index), smiles — one row per unique drug, giving you the raw string to tokenize for the Transformer, and separately, RDKit converts that same SMILES into an atom/bond graph for your GCN branch at runtime (this conversion typically happens in your data-loading code, not as a pre-saved file, since atom/bond graphs are usually built on-the-fly from SMILES using RDKit + PyTorch Geometric utilities).

What to do next
Join gnn_node_mapping.csv (drug_id, node_index) with your original drugs.csv (drug_id, smiles) → produces node_index, drug_id, smiles — this is your SMILES Transformer's actual input file
Keep gnn_edge_list_ready.csv separately — this is your DDI graph structure (likely feeding the cross-attention/interaction-level part of your architecture, or a separate DDI-graph-based branch if your design uses one, distinct from the per-molecule GCN)

Want me to write the join script for step 1 now, so you have your SMILES-ready file?




30 july owrk: 
joined gnn_node_mapping.csv (drug_id, node_index) with drugs.csv (drug_id, smiles) and Keep gnn_edge_list_ready.csv separately
- for transformer 

- ran biobtransformer : 
local vs repo- used local : and not using colab as sessions disconnect after 90 mins

- error when i ran on 10 drugs- tried on one- worked- 60 sec for one drug so 10-14 hours for 645. 

- biotransofrmer better than lagom- general chemical not pretrained on drugbank. biot is rule based- built for this task. 

- how to decide which metabolite" 
1. Iteration 1 over iteration 2+. BioTransformer runs multi-step cascades — iteration 1 = direct product of the parent, iteration 2 = metabolite-of-a-metabolite.- first pass better 
2. phase 1 vs phase 2 - phase two irreleavnt and excretion ready- one is ixidative.
3. structure similarity. 
4. enzyme probabiltyscores 



biot: column names
InChI / InChIKey – two ways of writing the metabolite's chemical structure as text; InChIKey is the short version, good for exact matching against other databases.
SMILES – another structure format, more commonly used for feeding into models/software.
PUBCHEM_CID – PubChem's ID for this compound, if it already exists in their database (helps check if the predicted metabolite is a known compound).
Molecular formula & Major Isotope Mass – basic chemistry info (atoms present, exact mass).
ALogP – predicted lipophilicity (how fat-soluble vs water-soluble the molecule is); relevant for how it behaves in the body.
Lipinski/Insecticide/Herbicide "Violations" – drug-likeness rule checks BioTransformer runs automatically; mostly not relevant to my project, can ignore.
Metabolite ID / cdk:Title – internal ID BioTransformer assigns to this specific metabolite.
Reaction – plain description of what happened to create this metabolite (e.g. "Alkyl-OH-glucuronidation," "Sulfation of secondary alcohol").
Reaction ID / Enzyme(s) – which specific enzyme(s) carried out that reaction (e.g. CYP1A2, CYP2A6).
Biosystem – confirms this was run in "HUMAN" mode.
Precursor ID / Precursor SMILES / Precursor InChI / Precursor InChIKey / Precursor ALogP / Precursor Mass – same info, but for the original drug this metabolite came from, so each row is self-contained (don't need to look anything up elsewhere to know parent vs. metabolite).


3 aug 

ok so : found that there are 700 rows for each file: how do we choose primary? 
1. theres primary and secondary(met of met) 
2. 700 rows bec it generates all possible- enzyme files, phase1 and 2, etc 
3. we want primary mets only 
4. ways i thought: 
- precursor id shows that if the smiles precursor is a parent then it is parent metabolite - we get all primary metabolites 
- then run structural similarity. 
or 
- see jiska precursor id hai remove it- then struct similarity 
or 
- try to map drugbank primary metabolite names to orignial smiles met csv- then just use that for each file . 
or
regrenerate files with layers=1 not 2 so that only primary is generated

going with precursor, for now 


