import unittest

import numpy as np
import torch

from baselines.models import MODEL_REGISTRY, build_model
from baselines.models.classification import InductiveHetMedGraphBuilder


class BaselineModelTests(unittest.TestCase):
    def test_required_segmentation_models_are_registered_and_preserve_resolution(self):
        x = torch.randn(2, 3, 64, 64)
        for name in ["unet", "attention_unet", "unetpp", "smu_net", "aau_net", "brn"]:
            with self.subTest(name=name):
                kwargs = {"backbone": "tiny"} if name == "brn" else {}
                model = build_model(name, task="segmentation", base_channels=8, **kwargs)
                oracle = {"oracle_mask": torch.ones(2, 1, 64, 64)} if name == "smu_net" else {}
                output = model(x, **oracle)
                logits = output["logits"] if isinstance(output, dict) else output
                self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))

    def test_required_classification_models_are_registered(self):
        self.assertIn("medsam_standard_fusion", MODEL_REGISTRY["classification"])
        self.assertIn("hetmed", MODEL_REGISTRY["classification"])

    def test_hetmed_validation_query_uses_training_reference_bank(self):
        model = build_model("hetmed", task="classification", image_dim=16, clinical_dim=5, num_classes=3, hidden_dim=8)
        query_img, query_clin = torch.randn(3, 16), torch.randn(3, 5)
        ref_img, ref_clin = torch.randn(7, 16), torch.randn(7, 5)
        builder = InductiveHetMedGraphBuilder(num_relations=4, threshold=0.75).fit(ref_clin.numpy())
        graph = builder.query_graph(query_clin.numpy(), ref_clin.numpy())
        output = model(query_img, query_clin, ref_img, ref_clin, graph)
        self.assertEqual(tuple(output["logits"].shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
