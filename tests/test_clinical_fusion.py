import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn

from bua_lel.models.bua_lel import BUALEL
from bua_lel.engine.trainer import TrainConfig, build_model


class _Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class _ForwardEncoder(nn.Module):
    def forward(self, image, return_dict=True):
        batch = image.shape[0]
        zero = image.new_zeros
        return {
            "seg_feats": (
                zero((batch, 120, 4, 4)), zero((batch, 240, 2, 2)),
                zero((batch, 480, 1, 1)), zero((batch, 960, 1, 1)),
            ),
            "cls_feat": zero((batch, 256, 4, 4)),
        }


class _Decoder(nn.Module):
    def forward(self, f1, f2, f3, f4):
        return f1


class _ROIPool(nn.Module):
    def forward(self, features, roi):
        batch = features.shape[0]
        zero = features.new_zeros
        return zero((batch, 256)), {
            "region_feats": zero((batch, 3, 256)),
            "region_weights": features.new_full((batch, 3), 1 / 3),
            "v_global": zero((batch, 256)),
        }


class _ClinicalGraph(nn.Module):
    def forward(self, clinical, mask):
        batch = clinical.shape[0]
        return {
            "clinical_nodes": clinical.new_zeros((batch, 12, 128)),
            "clinical_global": clinical.new_zeros((batch, 128)),
        }


class _HeteroGraph(nn.Module):
    def forward(self, morph_nodes, clinical_nodes, morph_global, clinical_global):
        return {"hetero_global": morph_nodes.new_zeros((morph_nodes.shape[0], 256))}


class _CaptureClassifier(nn.Module):
    def forward(self, features):
        self.last_input = features.detach().clone()
        return features.new_zeros((features.shape[0], 3))


def _build(**changes):
    kwargs = {
        "clinical_dim": 8,
        "architecture_profile": "paper",
        "modality": "multimodal",
        "fusion_mode": "heterog",
    }
    kwargs.update(changes)
    with patch("bua_lel.models.bua_lel.MedSAMMultiScaleEncoder", _Encoder):
        return BUALEL(**kwargs)


def _build_forward_model(**changes):
    changes.setdefault("late_raw_clinical_fusion", True)
    changes.setdefault("use_boundary_refiner", False)
    model = _build(**changes)
    model.encoder = _ForwardEncoder()
    model.decoder = _Decoder()
    model.roi_pool = _ROIPool()
    model.clin_graph = _ClinicalGraph()
    model.hetero_graph = _HeteroGraph()
    model.cls_head = _CaptureClassifier()
    return model


class LateRawClinicalFusionModelTests(unittest.TestCase):
    def test_default_keeps_graph_only_classifier_schema(self):
        model = _build()
        first_linear = next(module for module in model.cls_head.modules() if isinstance(module, nn.Linear))
        self.assertFalse(model.late_raw_clinical_fusion)
        self.assertEqual(first_linear.in_features, 256)

    def test_default_strictly_loads_the_graph_only_default_state_dict(self):
        graph_only_state = _build().state_dict()
        default_model = _build(late_raw_clinical_fusion=False)

        incompatible = default_model.load_state_dict(graph_only_state, strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            _build(late_raw_clinical_fusion=True).load_state_dict(
                graph_only_state, strict=True
            )

    def test_enabled_appends_raw_clinical_vector_to_classifier_input(self):
        model = _build(late_raw_clinical_fusion=True)
        fused = torch.randn(2, 256)
        clinical = torch.randn(2, 8)

        classifier_input = model.compose_classifier_input(fused, clinical)

        self.assertEqual(tuple(classifier_input.shape), (2, 264))
        torch.testing.assert_close(classifier_input[:, :256], fused)
        torch.testing.assert_close(classifier_input[:, 256:], clinical)
        first_linear = next(module for module in model.cls_head.modules() if isinstance(module, nn.Linear))
        self.assertEqual(first_linear.in_features, 264)

    def test_disabled_classifier_input_is_unchanged(self):
        model = _build(late_raw_clinical_fusion=False)
        fused = torch.randn(2, 256)
        clinical = torch.randn(2, 8)
        torch.testing.assert_close(model.compose_classifier_input(fused, clinical), fused)

    def test_late_clinical_fusion_rejects_ultrasound_only_modality(self):
        with self.assertRaisesRegex(ValueError, "late_raw_clinical_fusion"):
            _build(late_raw_clinical_fusion=True, modality="ultrasound_only")

    def test_main_forward_exposes_raw_clinical_tail_in_aux(self):
        model = _build_forward_model()
        image = torch.randn(2, 3, 16, 16)
        clinical = torch.randn(2, 8)

        _, logits, aux = model(image, clinical, torch.ones_like(clinical), task="both")

        self.assertEqual(tuple(logits.shape), (2, 3))
        torch.testing.assert_close(aux["classifier_input"][:, -8:], clinical)
        torch.testing.assert_close(model.cls_head.last_input, aux["classifier_input"])

    def test_region_perturbation_forward_uses_the_same_late_fusion(self):
        model = _build_forward_model()
        image = torch.randn(2, 3, 16, 16)
        clinical = torch.randn(2, 8)
        regions = torch.randn(2, 3, 256)

        logits = model.forward_cls_with_region_feats(
            image, clinical, torch.ones_like(clinical), regions
        )

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(model.cls_head.last_input.shape), (2, 264))
        torch.testing.assert_close(model.cls_head.last_input[:, -8:], clinical)

    def test_reduced_fusion_preserves_raw_aux_and_is_shared_by_both_paths(self):
        model = _build_forward_model(classifier_fused_dim=128)
        image = torch.randn(2, 3, 16, 16)
        clinical = torch.randn(2, 8)
        mask = torch.ones_like(clinical)

        _, logits, aux = model(image, clinical, mask, task="both")

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(aux["vfused"].shape), (2, 256))
        self.assertEqual(tuple(aux["vfused_raw"].shape), (2, 256))
        self.assertEqual(tuple(aux["classifier_fused"].shape), (2, 128))
        self.assertEqual(tuple(aux["classifier_input"].shape), (2, 136))
        torch.testing.assert_close(aux["vfused"], aux["vfused_raw"])
        torch.testing.assert_close(aux["classifier_input"][:, :128], aux["classifier_fused"])
        torch.testing.assert_close(aux["classifier_input"][:, -8:], clinical)

        regions = torch.randn(2, 3, 256)
        perturbed_logits = model.forward_cls_with_region_feats(
            image, clinical, mask, regions
        )
        self.assertEqual(tuple(perturbed_logits.shape), (2, 3))
        self.assertEqual(tuple(model.cls_head.last_input.shape), (2, 136))
        torch.testing.assert_close(model.cls_head.last_input[:, -8:], clinical)
