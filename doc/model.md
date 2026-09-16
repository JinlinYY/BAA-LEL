# Model and implementation

## Module correspondence

| Manuscript component | Implementation |
|---|---|
| BUA-LEL | `bua_lel/models/bua_lel.py:BUALEL` |
| Lesion-prior-aware semantic feature encoding | `backbones/medsam.py:MedSAMMultiScaleEncoder` |
| Anchor-constrained boundary graph | `boundary/anchor_boundary_graph.py:AnchorConstrainedBoundaryGraph` |
| Scale-adaptive zonal morphology encoding | `morphology/zonal_morphology.py:ZonalMorphologyEncoder` |
| Knowledge-guided clinical graph | `clinical/clinical_graph.py:ClinicalVariableGraphEncoder` |
| Morphology–clinical heterogeneous graph | `fusion/morph_clinical_graph.py:MorphClinicalHeteroGraph` |
| Segmentation and classification heads | `heads/segmentation.py`, `heads/classification.py` |

## Tensor interface

```python
from bua_lel.models import BUALEL

model = BUALEL(
    clinical_dim=12, num_classes=3,
    medsam_checkpoint_path="checkpoints/medsam_vit_b.pth",
    freeze_medsam=True, unfreeze_medsam_last_n=1,
    image_size=1024, lambda_anchor=0.01,
)
# image: float [B, 3, 256, 256], scaled to [0,1]
# clinical: float [B, clinical_dim], transformed using training-fold statistics
# observed: float tensor with the same shape; imputed values use ones
seg_logits, cls_logits, evidence = model(
    image, c_obs=clinical, m=observed, task="both"
)
```

The encoder resizes the image to its input grid; masks remain at the configured dataset resolution. The image encoder receives the loader's [0,1] tensor. `Sam.preprocess()` is not invoked by the BUA-LEL encoder wrapper, so its RGB pixel-mean normalization is not applied. Do not add another normalization step when reproducing this implementation.

The boundary ambiguity measure is `1 - abs(2 * P_low - 1)`. It is a foreground-probability ambiguity score, not an ensemble estimate of epistemic uncertainty. Boundary offsets contribute to anchor regularization; the final mask is predicted by dense refinement using graph context, not rasterized from displaced vertices.

The lesion prior is detached before region construction when `detach_roi_in_cls=True`. This blocks the direct classification gradient through morphology-zone construction; other shared features and boundary evidence still participate in joint learning.

## Clinical graph and fusion

The clinical encoder initializes **each of its 12 prior graph nodes with the complete clinical vector**, then applies graph convolution on a fixed knowledge-guided topology. Node labels describe prior graph positions; the implementation does not isolate one measured scalar per node. This distinction matters when interpreting node attention or adapting the graph to other cohorts. The topology is a modeling prior and does not establish causal effects.

The heterogeneous graph receives core, boundary, and peritumoral morphology nodes, geometry and ambiguity nodes, and clinical graph nodes. `late_raw_clinical_fusion` appends transformed clinical variables to the final graph readout before classification. This option is enabled in the default multimodal configurations and recorded in every checkpoint.

`architecture_profile: paper` constructs the multimodal graph path without unused alternative-fusion modules. `extended` constructs the single-modality and alternate-fusion paths. Both use the same scientific component implementations. Model parameter names are stable within this release; full checkpoints contain the configuration and clinical preprocessing needed to reconstruct the network.

## Objective

Joint training uses equal-weight BCE and Dice segmentation loss, class-weighted label-smoothed cross-entropy, task log-variance weighting, and an anchor regularizer with default coefficient 0.01. Dataset-specific configurations may override this objective. BUSI/ISIC2018 segmentation-only configurations exclude classification from the optimized loss and use Dice for checkpoint selection. Clinical-only ablation uses cross-entropy without segmentation supervision.

The three probability maps available for inspection are the coarse lesion prior (`seg_prob_low`), refined lesion prior (`roi_prob`), and boundary ambiguity (`boundary_uncertainty`). Evidence embeddings are descriptors, not pixelwise explanations.
