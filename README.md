# Pharmacon-project_X
# PHARMACON: 

An interpretable Drug-Drug Interaction (DDI) prediction system that integrates molecular structures with  biological pathway networks.


# 1. Overview of Project Approach
Traditional deep learning approaches for predicting polypharmacy side effects typically analyze only the parent drug structures given to a patient. PHARMACON is built to sit at the intersection of structural chemistry and systems biology by addressing a critical real-world gap: modeling both the parent drug and its primary metabolites, along with its biological context with protein interactions and pathways.

# 2. Core Principles
*   Four+ Three Stream Architecture: Simultaneously inputs Parent Drug A, Parent Drug B, Metabolite A, and Metabolite B, along with PPI, DPI and pathways .
*   Dual-Representation : Bridges local bond geometry (via Graph Convolutional Networks) with global sequence context (via SMILES Transformers).
*  Attribution-Driven Interpretability: Utilizes inter-molecular cross-attention to flag precise atom-level interaction sites, validating predictions against verified drugBank mechansims.

---

## 3. Weekly Goals & Status Tracker

| Week | Target Milestone | Status |
| :--- | :--- | :--- |
| **Week 1** | Learn from DL course (C1,C2,C5), understand the GNN paper. | Ongoing |
| **Week 2** | By sat(11 Jul)- find dataset compatibilty for all3, Wed: finish course2 + architecture diag | - |
| **Week 3** | - | - |
| **Week 4** | - | - |

---

### 4. Project Notes 
Navigate directly to the respective files for notes on the deep learning course, architectural strategy, and dataset curation status:

* **[Section 3: Deep Learning Course Notes](./notes/DLcourse_notes.md)**
* **[Section 4: Notes from Research & Domain Theory](./notes/researchNotes_theory.md)**
*  **[Section 5: Architectural Approach & Data Parsing Updates](./notes/architecture_and_parsing.md)**
* **[Section 6: Data Notes](./notes/data_notes.md)**

