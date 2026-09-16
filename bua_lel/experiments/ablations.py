"""Ablations."""
from dataclasses import dataclass, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class AblationConfig:
    group: str
    name: str
    medsam_mode: str = "freeze_all"
    modality: str = "multimodal"
    enabled_regions: Tuple[str, ...] = ("core", "boundary", "peritumoral")
    use_z_geo: bool = True
    use_z_unc: bool = True
    fusion_mode: str = "heterog"
    use_boundary_ambiguity: bool = True
    use_anchor_graph: bool = True
    use_zonal_evidence: bool = True
    model_family: str = "bua_lel"
    visual_backbone: str = "medsam_vit_b"
    backbone_unfreeze_policy: str = "last_transformer_block"


FULL_COMPONENT_MODEL = AblationConfig(
    group="components", name="full", medsam_mode="unfreeze_last_block"
)


def _spec(group: str, name: str, **changes) -> AblationConfig:
    return replace(FULL_COMPONENT_MODEL, group=group, name=name, **changes)


_REGIONS = {
    "core": ("core",),
    "boundary": ("boundary",),
    "peritumoral": ("peritumoral",),
    "core_boundary": ("core", "boundary"),
    "core_peritumoral": ("core", "peritumoral"),
    "boundary_peritumoral": ("boundary", "peritumoral"),
    "all_regions": ("core", "boundary", "peritumoral"),
}

ABLATIONS: Dict[str, Dict[str, AblationConfig]] = {
    "medsam": {
        "freeze_all": _spec("medsam", "freeze_all", medsam_mode="freeze_all"),
        "unfreeze_last_block": _spec("medsam", "unfreeze_last_block", medsam_mode="unfreeze_last_block"),
        "full_finetune": _spec("medsam", "full_finetune", medsam_mode="full_finetune"),
    },
    "modality": {
        name: _spec("modality", name, modality=name)
        for name in ("ultrasound_only", "clinical_only", "multimodal")
    },
    "regions": {
        name: _spec("regions", name, enabled_regions=regions)
        for name, regions in _REGIONS.items()
    },
    "boundary_tokens": {
        "none": _spec("boundary_tokens", "none", use_z_geo=False, use_z_unc=False),
        "geo_only": _spec("boundary_tokens", "geo_only", use_z_geo=True, use_z_unc=False),
        "unc_only": _spec("boundary_tokens", "unc_only", use_z_geo=False, use_z_unc=True),
        "both": _spec("boundary_tokens", "both"),
    },
    "fusion": {
        name: _spec("fusion", name, fusion_mode=name)
        for name in ("concat", "gate", "cross_attn", "heterog")
    },
    "components": {
        "full": FULL_COMPONENT_MODEL,
        "no_boundary_ambiguity": _spec("components", "no_boundary_ambiguity", use_boundary_ambiguity=False),
        "no_anchor_graph": _spec("components", "no_anchor_graph", use_anchor_graph=False),
        "no_zonal_evidence": _spec("components", "no_zonal_evidence", use_zonal_evidence=False),
        "no_heterogeneous_graph": _spec("components", "no_heterogeneous_graph", fusion_mode="concat"),
        "medsam_standard_decoder_fusion": _spec(
            "components", "medsam_standard_decoder_fusion",
            model_family="medsam_standard_decoder_fusion",
        ),
        "sam_vit_b_backbone": _spec(
            "components", "sam_vit_b_backbone",
            visual_backbone="sam_vit_b",
            backbone_unfreeze_policy="last_transformer_block",
        ),
        "imagenet_resnet50_backbone": _spec(
            "components", "imagenet_resnet50_backbone",
            visual_backbone="imagenet_resnet50",
            backbone_unfreeze_policy="last_bottleneck",
        ),
    },
}

ABLATION_GROUPS = {group: tuple(items) for group, items in ABLATIONS.items()}


FORMAL_ALIAS_SOURCES = {
    ("medsam", "unfreeze_last_block"): ("components", "full"),
    ("modality", "multimodal"): ("components", "full"),
    ("regions", "all_regions"): ("components", "full"),
    ("boundary_tokens", "both"): ("components", "full"),
    ("fusion", "concat"): ("components", "no_heterogeneous_graph"),
}

FORMAL_NAMED_ABLATIONS = (
    ("components", "full"),
    ("components", "no_boundary_ambiguity"),
    ("components", "no_anchor_graph"),
    ("components", "no_zonal_evidence"),
    ("components", "no_heterogeneous_graph"),
    ("fusion", "concat"),
    ("fusion", "gate"),
    ("fusion", "cross_attn"),
    ("modality", "ultrasound_only"),
    ("modality", "clinical_only"),
    ("modality", "multimodal"),
    ("boundary_tokens", "none"),
    ("boundary_tokens", "geo_only"),
    ("boundary_tokens", "unc_only"),
    ("boundary_tokens", "both"),
    ("regions", "core"),
    ("regions", "boundary"),
    ("regions", "peritumoral"),
    ("regions", "core_boundary"),
    ("regions", "core_peritumoral"),
    ("regions", "boundary_peritumoral"),
    ("regions", "all_regions"),
    ("medsam", "unfreeze_last_block"),
    ("medsam", "full_finetune"),
)

FORMAL_TRAINING_SOURCES = tuple(
    item for item in FORMAL_NAMED_ABLATIONS if item not in FORMAL_ALIAS_SOURCES
)


EXPLICIT_ONLY_ABLATIONS = (("medsam", "freeze_all"),)

assert len(FORMAL_NAMED_ABLATIONS) == 24
assert len(FORMAL_TRAINING_SOURCES) == 19
assert not set(EXPLICIT_ONLY_ABLATIONS) & set(FORMAL_NAMED_ABLATIONS)


def get_ablation(group: str, name: str) -> AblationConfig:
    try:
        return ABLATIONS[group][name]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation {group}/{name}") from exc
