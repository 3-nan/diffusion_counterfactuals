# Counterfactuals
Diffusion-based Counterfactual Generation
based on https://arxiv.org/pdf/2310.06668.pdf

Extension:
  - Addition of concept-based masking
  - Reduction of diffusion process on single concepts
  - Fun

1. run_ldce_baseline
      Implementation of the original LDCE method
2. run_concept_ldce
      Extension towards the restriction of the classifier gradient on single filters (concepts)
      based on <concept_layer> and <num_concepts>
3. other target
4. other target and concept restriction
5. concept and localization restriction
