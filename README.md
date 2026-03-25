# Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning

#  MIA Disparity & Unlearning Evaluation Framework

A lightweight, modular framework for evaluating **machine unlearning algorithms** using **multiple Membership Inference Attacks (MIAs)** and **ensemble-based privacy analysis**.

This project builds on two external repositories:

-  MIA implementations & disparity framework:  
  https://github.com/RPI-DSPlab/mia-disparity

-  Unlearning algorithms (SCRUB, SSD, BadTeacher, Amnesiac):  
  https://github.com/OngWinKent/MachineUnlearning

---

#  Research Goal

Quantify **disparities among MIAs** and evaluate **privacy risks after unlearning**.

We aim to:

- Understand how different MIAs detect **different subsets of vulnerable samples**
- Measure **stability** of attacks across seeds
- Build **ensembles of MIAs** to improve coverage and robustness
- Provide a more **comprehensive privacy evaluation** for unlearning methods

---

#  Key Concepts

### Disparity
- **Coverage**: how many unique training samples are flagged as members  
- **Stability**: how consistent predictions are across seeds/runs  

### Evaluation Targets
- `forget_vs_test` → primary privacy evaluation  
- `retain_vs_test` → utility vs privacy  
- `forget_vs_retain` → diagnostic (not standard MIA)  

### Ensembles
- **OR (Union)** → maximize coverage  
- **k-of-M Voting** → tradeoff between precision and recall  