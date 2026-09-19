from typing import Any, Dict

from baa_lel.models.baa_lel import BAALEL
from baa_lel.models.medsam_mtl import MedSAMMTLModel
from baselines.models.classification import HetMedClassifier, MedSAMStandardFusion
from baselines.models.official_classification import (
    AMFMedITClassifier,
    HyperFusionClassifier,
    KMNetClassifier,
    MAPClassifier,
    MIINetClassifier,
)
from baselines.models.multitask import MTANet
from baselines.models.segmentation import AAUNet, AttentionUNet, BRN, MedSAMStandardDecoder, SMUNet, TransUNet, UNet, UNetPlusPlus


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "segmentation": {
        "unet": UNet,
        "attention_unet": AttentionUNet,
        "unetpp": UNetPlusPlus,
        "transunet": TransUNet,
        "smu_net": SMUNet,
        "aau_net": AAUNet,
        "brn": BRN,
        "medsam_standard_decoder": MedSAMStandardDecoder,
    },
    "classification": {
        "medsam_standard_fusion": MedSAMStandardFusion,
        "hetmed": HetMedClassifier,
        "miinet": MIINetClassifier,
        "kmnet": KMNetClassifier,
        "hyperfusion": HyperFusionClassifier,
        "amf_medit": AMFMedITClassifier,
        "map": MAPClassifier,
    },
    "multitask": {
        "mtanet": MTANet,
        "baa_lel": BAALEL,
        "medsam_mtl": MedSAMMTLModel,
    },
}


REFERENCE_BASELINE_NAMES = frozenset()

MODEL_METADATA = {
    "unet": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "canonical_local", "source": "local",
        "version": "baa-lel-unet-v1", "upstream_version": "canonical", "display_name": "U-Net",
    },
    "attention_unet": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "canonical_local", "source": "local",
        "version": "baa-lel-attention-unet-v1", "upstream_version": "canonical", "display_name": "Attention U-Net",
    },
    "unetpp": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "canonical_local", "source": "local",
        "version": "baa-lel-unetpp-v1", "upstream_version": "canonical", "display_name": "U-Net++",
    },
    "transunet": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "author_source_informed_adapter",
        "source": "https://github.com/Beckschen/TransUNet",
        "version": "baa-lel-transunet-v1", "upstream_version": "02ef0010b36eb8328b5e689eadaf613602edf9b8",
        "display_name": "TransUNet",
    },
    "smu_net": {
        "task": "segmentation", "interactive": True,
        "eligible_for_automatic_comparison": False,
        "implementation": "paper_reimplementation",
        "source": "https://doi.org/10.1109/TMI.2021.3116087", "version": "baa-lel-smu-v1",
        "upstream_version": "paper-2021",
        "display_name": "SMU-Net (3-click oracle)",
    },
    "aau_net": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "author_source_informed_port",
        "source": "https://github.com/CGPxy/AAU-net",
        "version": "baa-lel-aau-v1", "upstream_version": "0cf0121566a09cdd229e1bd57cac5318718d871a",
        "display_name": "AAU-Net",
    },
    "brn": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "paper_reimplementation",
        "source": "https://doi.org/10.1016/j.media.2022.102478", "version": "baa-lel-brn-v2",
        "upstream_version": "paper-2022-resnet101-assumption",
        "display_name": "BRN",
    },
    "medsam_standard_decoder": {
        "task": "segmentation", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "local_medsam_encoder_standard_decoder",
        "source": "local", "version": "baa-lel-medsam-decoder-v2", "upstream_version": "local-medsam",
        "display_name": "MedSAM + standard decoder",
    },
    "medsam_standard_fusion": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "local_medsam_encoder_standard_concat_fusion",
        "source": "local", "version": "baa-lel-medsam-fusion-v2", "upstream_version": "local-medsam",
        "display_name": "MedSAM + standard fusion",
    },
    "hetmed": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "author_source_informed_port",
        "source": "https://github.com/Sein-Kim/Multimodal-Medical",
        "version": "baa-lel-hetmed-v5", "upstream_version": "c235485673e3f6040ab301c312b17bc62ef720d6",
        "display_name": "HetMed",
    },
    "miinet": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "author_source_informed_leakage_safe_port",
        "source": "https://github.com/JinlinYY/MIINet",
        "version": "baa-lel-miinet-v1", "upstream_version": "239074d9ab8b9c6879b07f9b96eee8200154a877",
        "display_name": "MIINet",
    },
    "kmnet": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "author_source_informed_leakage_safe_port",
        "source": "https://github.com/JinlinYY/KMNet",
        "version": "baa-lel-kmnet-v1", "upstream_version": "main-audited-2026-09-02",
        "display_name": "KMNet",
    },
    "hyperfusion": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "independent_source_informed_port",
        "source": "https://github.com/daniel4725/HyperFusion",
        "version": "baa-lel-hyperfusion-v3", "upstream_version": "96a102c766cbdd97882cab485cf063050a2adfa1",
        "display_name": "HyperFusion",
    },
    "amf_medit": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "author_source_informed_adapter",
        "source": "https://github.com/Jasmine-ycj/AMF-MedIT",
        "version": "baa-lel-amf-medit-v3", "upstream_version": "6dfa7b6d366529c34c8064486e65ca1a77dc91dc",
        "display_name": "AMF-MedIT",
    },
    "map": {
        "task": "classification", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "independent_source_informed_port",
        "source": "https://github.com/ZhangJD-ong/HER2-MAP-from-Multimodal-Breast-Data",
        "version": "baa-lel-map-v3", "upstream_version": "9e07013568733b08a8aeda31660a1ea725b6a7d5",
        "display_name": "MAP",
    },
    "mtanet": {
        "task": "multitask", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "paper_reimplementation",
        "source": "https://doi.org/10.1109/TMI.2023.3317088",
        "version": "baa-lel-mtanet-v1", "upstream_version": "paper-2024",
        "display_name": "MTANet",
    },
    "baa_lel": {
        "task": "multitask", "interactive": False,
        "eligible_for_automatic_comparison": False,
        "implementation": "baa_lel", "source": "local",
        "version": "baa-lel-v1", "upstream_version": "not-applicable",
        "display_name": "BAA-LEL",
    },
    "medsam_mtl": {
        "task": "multitask", "interactive": False,
        "eligible_for_automatic_comparison": True,
        "implementation": "local_shared_medsam_plain_multitask_baseline", "source": "local",
        "version": "baa-lel-medsam-mtl-v1", "upstream_version": "local-medsam",
        "display_name": "MedSAM-MTL",
    },
}


def build_model(name: str, task: str, **kwargs):
    try:
        constructor = MODEL_REGISTRY[task][name]
    except KeyError as exc:
        available = sorted(MODEL_REGISTRY.get(task, {}))
        raise ValueError(f"unknown {task} model {name!r}; available={available}") from exc
    return constructor(**kwargs)
