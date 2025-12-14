# Weight Perturbation Pipeline

1. Run `adaptive_attack.py` to fine-tune the child models and produce the adversarial models.
2. Run `gen_feas_Calt_adaptive.py` to extract the corresponding discriminative features.
3. Use `main_Calt1213.py` to perform the final attestation (verification).

Got it—if you’re running other attack types, make sure to move the corresponding attack scripts/functions into this current directory before executing them.

