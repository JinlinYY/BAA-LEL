import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from bua_lel.models.backbones.medsam import MedSAMMultiScaleEncoder


class _PatchEmbed(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.5))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tanh(x * self.weight)


class _ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.pos_embed = None
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.neck = nn.Identity()


def _run(use_checkpointing):
    encoder = _ImageEncoder()
    wrapper = SimpleNamespace(
        image_encoder=encoder,
        out_indices=(0, 1),
        activation_checkpointing=use_checkpointing,
        training=True,
    )
    image = torch.randn(2, 3, 4, 4, requires_grad=True)
    output, intermediate = MedSAMMultiScaleEncoder.forward_medsam_with_intermediate(wrapper, image)
    loss = output.square().mean() + sum(value.square().mean() for value in intermediate.values())
    loss.backward()
    return (
        output.detach(), image.grad.detach(),
        [block.weight.grad.detach() for block in encoder.blocks],
        [block.calls for block in encoder.blocks],
    )


class MedSAMActivationCheckpointingTests(unittest.TestCase):
    def test_checkpointing_preserves_forward_and_gradients(self):
        torch.manual_seed(7)
        plain = _run(False)
        torch.manual_seed(7)
        checkpointed = _run(True)
        torch.testing.assert_close(plain[0], checkpointed[0])
        torch.testing.assert_close(plain[1], checkpointed[1])
        for expected, observed in zip(plain[2], checkpointed[2]):
            torch.testing.assert_close(expected, observed)
        self.assertEqual(plain[3], [1, 1])
        self.assertTrue(all(observed > expected for expected, observed in zip(plain[3], checkpointed[3])))


if __name__ == "__main__":
    unittest.main()
