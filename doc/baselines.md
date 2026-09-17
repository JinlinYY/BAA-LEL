# Baseline methods

The repository includes baseline implementations and adaptations. The model registry and its metadata are in `baselines/models/__init__.py`.

| Method | Task | Implementation and assumptions |
|---|---|---|
| U-Net, U-Net++ | Segmentation | Local encoder-decoder implementations |
| TransUNet | Segmentation | Source-informed 2D hybrid CNN/Transformer adapter |
| AAU-Net | Segmentation | Independent PyTorch adaptation of HAAM attention modules |
| BRN | Segmentation | Paper-based boundary rendering implementation with ResNet-101 assumption |
| SMU-Net | Segmentation | Paper-based morphology stream with three reference-mask-derived clicks; oracle-assisted reference |
| MedSAM-Seg | Segmentation | MedSAM encoder + standard decoder (`medsam_standard_decoder`) |
| MIINet | Classification | Source-informed image–clinical interaction; fold-local preprocessing |
| KMNet | Classification | Source-informed visual and medical-prior clinical graph adaptation |
| HyperFusion | Classification | Independent 2D adaptation of clinical-conditioned image convolutions |
| AMF-MedIT | Classification | Source-informed align-modulation-fusion; requires `mamba-ssm` for research training |
| MAP | Classification | Adaptation using the available imaging and clinical modalities |
| HetMed | Classification | Inductive patient-graph adaptation; queries connect only to training references |
| MTANet | Joint | Paper-based multi-task attention and image–clinical bottleneck |
| MedSAM-MTL | Joint | Shared MedSAM encoder, segmentation decoder, global image–clinical fusion |

SMU-Net is an oracle-assisted reference using clicks derived from reference masks. HetMed supports ImageNet/ResNet-50 initialization or a supplied checkpoint.

## Running

```bash
python -m pip install -r requirements-baselines.txt
python scripts/run_baseline_cv.py --dataset HER2USC --method unet \
  --task segmentation --data-root /path/to/source --output-root outputs/baselines
python scripts/run_baseline_cv.py --dataset HER2USC --method miinet \
  --task classification --data-root /path/to/source --output-root outputs/baselines
python scripts/run_multitask_cv.py --dataset HER2USC --method mtanet \
  --data-root /path/to/source --output-root outputs/baselines
python scripts/run_hetmed_cv.py --help
```

The method keys are `unet`, `unetpp`, `transunet`, `aau_net`, `brn`, `smu_net`, `medsam_standard_decoder`, `miinet`, `kmnet`, `hyperfusion`, `amf_medit`, `map`, and `hetmed`. Attention U-Net and MedSAM standard fusion are available as additional controls. HetMed uses its dedicated runner for patient-graph training. Consult each runner's `--help` for model-specific parameters.

AMF-MedIT requires a compatible Linux/CUDA `mamba-ssm` installation. Install PyTorch first, then install `mamba-ssm` following that project's build requirements. SMU-Net's LSC superpixels require OpenCV contrib; do not install conflicting OpenCV distributions in the same environment.

Nested runners share one outer patient manifest per dataset and fit preprocessing on training subsets. Their default epoch count is 50; use `--epochs` and other explicit options to declare your comparison protocol. Source layouts are described in [data.md](data.md).

## Sources

Architecture references retained in the registry include [TransUNet](https://github.com/Beckschen/TransUNet), [AAU-Net](https://github.com/CGPxy/AAU-net), [HetMed](https://github.com/Sein-Kim/Multimodal-Medical), [MIINet](https://github.com/JinlinYY/MIINet), [KMNet](https://github.com/JinlinYY/KMNet), [HyperFusion](https://github.com/daniel4725/HyperFusion), [AMF-MedIT](https://github.com/Jasmine-ycj/AMF-MedIT), and [MAP](https://github.com/ZhangJD-ong/HER2-MAP-from-Multimodal-Breast-Data). Registry entries record source revisions where available. External source links identify architectural references; they do not extend the MIT license to those repositories or their datasets.
