"""Paired, ultrasound-safe image and mask augmentation."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_training_transform(profile: str):
    """Build a training-only paired transform from its stable config name."""
    if profile == "none":
        return None
    if profile == "ultrasound_mild":
        return PairedUltrasoundAugment()
    if profile == "ultrasound_gentle":
        return PairedUltrasoundAugment(
            affine_probability=0.6,
            intensity_probability=0.15,
            noise_probability=0.10,
            max_rotation_degrees=8.0,
            max_translation_fraction=0.03,
            scale_range=(0.95, 1.05),
            intensity_range=(0.95, 1.05),
            max_noise_std=0.01,
        )
    if profile == "dermoscopy_mild":
        return PairedDermoscopyAugment()
    raise ValueError(f"unknown augmentation profile: {profile!r}")


class PairedDermoscopyAugment:
    """Dihedral geometry plus conservative RGB photometric jitter."""

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotate_probability: float = 0.75,
        color_probability: float = 0.5,
        brightness_delta: float = 0.08,
        contrast_delta: float = 0.10,
    ) -> None:
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.rotate_probability = float(rotate_probability)
        self.color_probability = float(color_probability)
        self.brightness_delta = float(brightness_delta)
        self.contrast_delta = float(contrast_delta)
        probabilities = (
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
            self.rotate_probability,
            self.color_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if image.dim() != 3 or mask.dim() != 3 or image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("paired augmentation expects aligned [C,H,W] image and mask")
        if torch.rand(()) < self.horizontal_flip_probability:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(()) < self.vertical_flip_probability:
            image, mask = image.flip(-2), mask.flip(-2)
        if torch.rand(()) < self.rotate_probability:
            turns = int(torch.randint(0, 4, ()).item())
            image, mask = torch.rot90(image, turns, (-2, -1)), torch.rot90(mask, turns, (-2, -1))
        if torch.rand(()) < self.color_probability:
            brightness = torch.empty(()).uniform_(-self.brightness_delta, self.brightness_delta)
            contrast = torch.empty(()).uniform_(1.0 - self.contrast_delta, 1.0 + self.contrast_delta)
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (image - mean) * contrast + mean + brightness
        return image.clamp(0.0, 1.0), (mask > 0.5).to(mask.dtype)


class PairedUltrasoundAugment:
    """Apply mild geometry to image/mask and intensity changes to image only."""

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
        affine_probability: float = 0.7,
        intensity_probability: float = 0.3,
        noise_probability: float = 0.3,
        max_rotation_degrees: float = 10.0,
        max_translation_fraction: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
        intensity_range: tuple[float, float] = (0.9, 1.1),
        max_noise_std: float = 0.02,
    ) -> None:
        probabilities = (
            horizontal_flip_probability,
            affine_probability,
            intensity_probability,
            noise_probability,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError("scale_range must be positive and ordered")
        if intensity_range[0] <= 0 or intensity_range[0] > intensity_range[1]:
            raise ValueError("intensity_range must be positive and ordered")
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.affine_probability = float(affine_probability)
        self.intensity_probability = float(intensity_probability)
        self.noise_probability = float(noise_probability)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.max_translation_fraction = float(max_translation_fraction)
        self.scale_range = tuple(map(float, scale_range))
        self.intensity_range = tuple(map(float, intensity_range))
        self.max_noise_std = float(max_noise_std)

    @staticmethod
    def _uniform(low: float, high: float, reference: torch.Tensor) -> torch.Tensor:
        return torch.empty((), device=reference.device).uniform_(low, high)

    def _affine(self, image: torch.Tensor, mask: torch.Tensor):
        angle = self._uniform(
            -self.max_rotation_degrees, self.max_rotation_degrees, image
        ) * (math.pi / 180.0)
        scale = self._uniform(self.scale_range[0], self.scale_range[1], image)
        tx = self._uniform(
            -2.0 * self.max_translation_fraction,
            2.0 * self.max_translation_fraction,
            image,
        )
        ty = self._uniform(
            -2.0 * self.max_translation_fraction,
            2.0 * self.max_translation_fraction,
            image,
        )
        cosine, sine = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack([
            torch.stack([cosine, -sine, tx]),
            torch.stack([sine, cosine, ty]),
        ]).to(dtype=image.dtype).unsqueeze(0)
        image_batch, mask_batch = image.unsqueeze(0), mask.unsqueeze(0)
        grid = F.affine_grid(theta, image_batch.shape, align_corners=False)
        image = F.grid_sample(
            image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).squeeze(0)
        mask = F.grid_sample(
            mask_batch, grid, mode="nearest", padding_mode="zeros", align_corners=False
        ).squeeze(0)
        return image, mask

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if image.dim() != 3 or mask.dim() != 3 or image.shape[-2:] != mask.shape[-2:]:
            raise ValueError("paired augmentation expects aligned [C,H,W] image and mask")
        if torch.rand((), device=image.device) < self.horizontal_flip_probability:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand((), device=image.device) < self.affine_probability:
            image, mask = self._affine(image, mask)
        if torch.rand((), device=image.device) < self.intensity_probability:
            contrast = self._uniform(self.intensity_range[0], self.intensity_range[1], image)
            gamma = self._uniform(self.intensity_range[0], self.intensity_range[1], image)
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = ((image - mean) * contrast + mean).clamp(0.0, 1.0).pow(gamma)
        if torch.rand((), device=image.device) < self.noise_probability:
            noise_std = self._uniform(0.0, self.max_noise_std, image)


            noise = torch.randn_like(image[:1]) * noise_std
            image = image + noise
        return image.clamp(0.0, 1.0), (mask > 0.5).to(mask.dtype)
