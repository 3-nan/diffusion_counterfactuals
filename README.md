# Concept-Localized Diffusion Counterfactuals
 
**What would need to change in this image for the classifier to decide differently, and only in the parts that matter?**
 
This repo generates visual counterfactual explanations for image classifiers using latent diffusion. It builds on [LDCE (Latent Diffusion Counterfactual Explanations)](https://arxiv.org/abs/2310.06668) and adds **concept-based masking**, so edits are restricted to human-interpretable concepts and localized regions instead of altering the whole image.
 
<!-- Replace with a real figure: original | LDCE baseline | concept-localized (ours) -->
<p align="center">
  <img src="assets/teaser.png" width="800" alt="Original image, LDCE baseline counterfactual, and concept-localized counterfactual side by side">
</p>
## Why it matters
Standard counterfactuals often change many things at once, which makes them hard to read. Restricting the classifier gradient to a chosen concept, and running diffusion only where that concept lives, produces edits that are **smaller, targeted and easier to interpret**. This is relevant for safety-critical vision systems such as autonomous driving.
 
## Method in brief
1. **Baseline:** LDCE guides a latent diffusion model with classifier gradients toward a target class.
2. **Concept-restricted guidance:** classifier gradients are filtered so only a selected concept contributes.
3. **Localized diffusion:** the denoising process is masked to the concept's spatial region.
## Quickstart
```bash
git clone https://github.com/3-nan/diffusion_counterfactuals.git
cd diffusion_counterfactuals
docker build -t dce -f Dockerfile .
docker run --gpus all -it dce python run_concept_ldce.py --config=configs/ldce/v1_concept.yaml
```
 
## Entry points
| Script | Purpose |
|---|---|
| `run_ldce_baseline` | Original LDCE method |
| `run_concept_ldce` | Concept-restricted classifier-gradient filtering |
| _[other variants]_ | Target specification, concept + localization restrictions |
| _[eval script]_ | Evaluation of generated counterfactuals |
 
Dockerfiles are provided for training, CLIP and diffusers environments.
 
## Results
<!-- One small table beats a paragraph. Example columns: method | validity | proximity / LPIPS | edit area | FID -->
| Method | _[metric 1]_ | _[metric 2]_ | _[metric 3]_ |
|---|---|---|---|
| LDCE (baseline) | | | |
| Concept-localized (ours) | | | |
 
## Citation
If you use this code, please cite:
```bibtex
@inproceedings{motzkus_coladce_2025,
  title={Concepts Guide and Explain Diffusion Visual Counterfactuals.},
  author={Motzkus, Franz and Schmid, Ute},
  booktitle={xAI (Late-breaking Work, Demos, Doctoral Consortium)},
  pages={177--184},
  year={2025}
}
```
And the original LDCE paper: Farid et al., *Latent Diffusion Counterfactual Explanations*, https://arxiv.org/pdf/2310.06668.pdf, 2023.

