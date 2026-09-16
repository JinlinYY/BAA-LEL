"""Comparative segmentation baselines.

SMU-Net and BRN are independent paper reimplementations. AAU-Net is an
independent PyTorch port informed by the authors' TensorFlow HAAM module. No
source from the unlicensed author repositories is vendored here.
"""
from __future__ import annotations

import math
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return max(groups, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetCore(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        channels = [base_channels * 2**i for i in range(4)]
        self.encoders = nn.ModuleList(
            [ConvBlock(in_channels, channels[0])]
            + [ConvBlock(channels[i - 1], channels[i]) for i in range(1, 4)]
        )
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.decoders = nn.ModuleList(
            [ConvBlock(channels[i] * 3, channels[i]) for i in reversed(range(4))]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        decoded = []
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], 1))
            decoded.append(x)
        return decoded


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, **_: object):
        super().__init__()
        self.core, self.head = UNetCore(in_channels, base_channels), nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(x)[-1])


class AttentionUNet(UNet):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, **kwargs: object):
        super().__init__(in_channels, base_channels, **kwargs)
        self.gate = nn.Sequential(nn.Conv2d(base_channels, base_channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.core(x)[-1]
        return self.head(feature * self.gate(feature))


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, **_: object):
        super().__init__()
        c = base_channels
        self.x00, self.x10, self.x20 = ConvBlock(in_channels, c), ConvBlock(c, 2*c), ConvBlock(2*c, 4*c)
        self.x01, self.x11, self.x02 = ConvBlock(3*c, c), ConvBlock(6*c, 2*c), ConvBlock(4*c, c)
        self.pool, self.head = nn.MaxPool2d(2), nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x); x10 = self.x10(self.pool(x00)); x20 = self.x20(self.pool(x10))
        x01 = self.x01(torch.cat([x00, F.interpolate(x10, x00.shape[-2:], mode="bilinear", align_corners=False)], 1))
        x11 = self.x11(torch.cat([x10, F.interpolate(x20, x10.shape[-2:], mode="bilinear", align_corners=False)], 1))
        x02 = self.x02(torch.cat([x00, x01, F.interpolate(x11, x00.shape[-2:], mode="bilinear", align_corners=False)], 1))
        return self.head(x02)


class TransUNet(nn.Module):
    """TransUNet."""
    def __init__(self, in_channels: int = 3, base_channels: int = 32, transformer_dim: int = 256, transformer_layers: int = 4, **_: object):
        super().__init__()
        c = int(base_channels)
        self.stem = ConvBlock(in_channels, c)
        self.encoder1 = ConvBlock(c, 2 * c)
        self.encoder2 = ConvBlock(2 * c, 4 * c)
        self.encoder3 = ConvBlock(4 * c, transformer_dim)
        self.pool = nn.MaxPool2d(2)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=8, dim_feedforward=4 * transformer_dim,
            dropout=.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.decode3 = ConvBlock(transformer_dim + 4 * c, 4 * c)
        self.decode2 = ConvBlock(4 * c + 2 * c, 2 * c)
        self.decode1 = ConvBlock(2 * c + c, c)
        self.decode0 = ConvBlock(c + c, c)
        self.head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _position(height: int, width: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij",
        )
        frequencies = torch.arange(channels // 4, device=device, dtype=dtype).view(1, 1, -1) + 1
        return torch.cat([
            torch.sin(xx[..., None] * frequencies), torch.cos(xx[..., None] * frequencies),
            torch.sin(yy[..., None] * frequencies), torch.cos(yy[..., None] * frequencies),
        ], dim=-1).view(1, height * width, -1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skip0 = self.stem(image)
        skip1 = self.encoder1(self.pool(skip0))
        skip2 = self.encoder2(self.pool(skip1))
        encoded = self.encoder3(self.pool(skip2))
        encoded = self.pool(encoded)
        batch, channels, height, width = encoded.shape
        tokens = encoded.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens + self._position(height, width, channels, image.device, image.dtype))
        decoded = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        decoded = self.decode3(torch.cat([F.interpolate(decoded, skip2.shape[-2:], mode="bilinear", align_corners=False), skip2], 1))
        decoded = self.decode2(torch.cat([F.interpolate(decoded, skip1.shape[-2:], mode="bilinear", align_corners=False), skip1], 1))
        decoded = self.decode1(torch.cat([F.interpolate(decoded, skip0.shape[-2:], mode="bilinear", align_corners=False), skip0], 1))
        decoded = self.decode0(torch.cat([decoded, skip0], 1))
        return self.head(decoded)


def simulate_three_clicks(mask: torch.Tensor) -> torch.Tensor:
    """Deterministic lesion clicks in (y, x): centroid, then farthest points."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    height, width = mask.shape[-2:]
    batches = []
    for sample in mask[:, 0] > 0.5:
        coordinates = torch.nonzero(sample, as_tuple=False)
        if not coordinates.numel():
            centre = torch.tensor([height // 2, width // 2], device=mask.device)
            batches.append(centre.repeat(3, 1)); continue
        values = coordinates.float()
        first = ((values - values.mean(0, keepdim=True)) ** 2).sum(1).argmin()
        selected = [coordinates[first]]
        for _ in range(2):
            distance = torch.cdist(values, torch.stack(selected).float()).amin(1)
            selected.append(coordinates[distance.argmax()])
        batches.append(torch.stack(selected))
    return torch.stack(batches)


class SMUSaliencyGenerator:
    """Three-scale LSC saliency preprocessing for the SMU-Net oracle.

    OpenCV contrib provides the paper's LSC algorithm. ``strict_lsc=False`` is
    reserved for CPU synthetic tests and falls back to SLIC with the same region
    counts; formal benchmark runners enable strict mode.
    """
    cluster_counts = (8, 15, 50)

    def __init__(self, strict_lsc: bool = True, cache_size: int = 1024, cache_dir: Optional[str] = None, parallel_workers: int = 4):
        self.strict_lsc = bool(strict_lsc)
        self.cache_size = max(0, int(cache_size))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_workers = max(1, int(parallel_workers))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def _cache_key(image: np.ndarray, points: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(image, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(points, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def _segments(self, image: np.ndarray, count: int) -> np.ndarray:
        import cv2
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelLSC"):
            region_size = max(2, int(round(math.sqrt(image.shape[0]*image.shape[1]/count))))
            algorithm = cv2.ximgproc.createSuperpixelLSC((image*255).astype(np.uint8), region_size=region_size, ratio=.075)
            algorithm.iterate(10); algorithm.enforceLabelConnectivity(min_element_size=10)
            return algorithm.getLabels().astype(np.int32)
        if self.strict_lsc:
            raise RuntimeError("SMU-Net LSC preprocessing requires opencv-contrib-python-headless")
        from skimage.segmentation import slic
        return slic(image, n_segments=count, compactness=10.0, start_label=0, channel_axis=-1).astype(np.int32)

    @staticmethod
    def _region_saliency(image: np.ndarray, labels: np.ndarray, points: np.ndarray) -> np.ndarray:
        height, width = labels.shape; region_ids = np.unique(labels)
        colours, centres = [], []
        for region_id in region_ids:
            yy, xx = np.nonzero(labels == region_id)
            colours.append(image[yy, xx].mean(0)); centres.append([yy.mean()/max(height-1,1), xx.mean()/max(width-1,1)])
        colours, centres = np.asarray(colours), np.asarray(centres)
        clicked_ids = np.unique(labels[points[:, 0], points[:, 1]])
        clicked_indices = [int(np.flatnonzero(region_ids == value)[0]) for value in clicked_ids]
        prototype = colours[clicked_indices].mean(0)
        colour_distance = np.linalg.norm(colours-prototype, axis=1)
        colour_scale = max(float(np.median(colour_distance[colour_distance > 0])) if np.any(colour_distance > 0) else 1.0, 1e-6)
        normalized_points = points.astype(np.float32) / np.asarray([[max(height-1,1), max(width-1,1)]], np.float32)
        spatial_distance = np.linalg.norm(centres[:, None, :]-normalized_points[None, :, :], axis=-1).min(1)
        scores = np.exp(-colour_distance/colour_scale) * np.exp(-spatial_distance/.5)
        output = np.zeros(labels.shape, np.float32)
        for region_id, score in zip(region_ids, scores): output[labels == region_id] = score
        return output

    def _compute_foreground(self, sample: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
        import cv2
        scale_maps = [self._region_saliency(sample, self._segments(sample, count), sample_points) for count in self.cluster_counts]
        low_level = np.mean(scale_maps, axis=0)
        high_level = cv2.GaussianBlur(low_level, (0, 0), sigmaX=max(sample.shape[:2])/32)
        foreground = ((2*low_level+high_level)/3).astype(np.float32, copy=False)
        foreground -= foreground.min(); foreground /= max(float(foreground.max()), 1e-6)
        return foreground

    def _disk_path(self, key: str) -> Optional[Path]:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.npy"

    def _load_disk(self, key: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return None
        try:
            value = np.load(path, allow_pickle=False)
            if value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all():
                return value
        except (OSError, ValueError):
            pass
        return None

    def _save_disk(self, key: str, value: np.ndarray) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        temporary = path.with_name(f".{path.stem}.tmp.npy")
        np.save(temporary, value, allow_pickle=False)
        temporary.replace(path)

    def __call__(self, image: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3: raise ValueError("image must have shape [B,3,H,W]")
        images = image.detach().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
        point_values = points.detach().cpu().numpy().astype(np.int64)
        outputs: List[Optional[np.ndarray]] = [None] * len(images)
        missing = []
        for index, (sample, sample_points) in enumerate(zip(images, point_values)):
            key = self._cache_key(sample, sample_points)
            foreground = self._cache.get(key)
            if foreground is None:
                foreground = self._load_disk(key, sample.shape[:2])
            if foreground is None:
                missing.append((index, key, sample, sample_points))
            else:
                outputs[index] = foreground
                if self.cache_size:
                    self._cache[key] = foreground
                    self._cache.move_to_end(key)
        if missing:
            workers = min(self.parallel_workers, len(missing))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                computed = list(executor.map(lambda item: self._compute_foreground(item[2], item[3]), missing))
            for (index, key, _, _), foreground in zip(missing, computed):
                outputs[index] = foreground
                self._save_disk(key, foreground)
                if self.cache_size:
                    self._cache[key] = foreground.copy()
                    self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        foreground = torch.from_numpy(np.stack(outputs)[:, None]).to(device=image.device, dtype=image.dtype)
        return foreground, foreground.amax((2, 3), keepdim=True)-foreground


def _point_guidance(image: torch.Tensor, points: torch.Tensor):
    batch, _, height, width = image.shape
    yy = torch.linspace(0, 1, height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    xx = torch.linspace(0, 1, width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
    py = points[..., 0].to(image.dtype) / max(height - 1, 1)
    px = points[..., 1].to(image.dtype) / max(width - 1, 1)
    distance = (yy - py[:, :, None, None]) ** 2 + (xx - px[:, :, None, None]) ** 2
    foreground = torch.exp(-distance / (2 * 0.12**2)).amax(1, keepdim=True)
    grey = image.mean(1, keepdim=True)
    sampled = torch.stack([grey[i, 0, points[i, :, 0], points[i, :, 1]].mean() for i in range(batch)])
    foreground = (2 * foreground + foreground * torch.exp(-torch.abs(grey - sampled[:, None, None, None]) / .25)) / 3
    background = 1 - foreground
    sobel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    edge = torch.sqrt(F.conv2d(foreground, sobel_x, padding=1).square() + F.conv2d(foreground, sobel_y, padding=1).square() + 1e-8).clamp(0, 1)
    cy, cx = py.mean(1)[:, None, None, None], px.mean(1)[:, None, None, None]
    position = 1 - torch.sqrt((yy-cy).square() + (xx-cx).square()).clamp(max=math.sqrt(2)) / math.sqrt(2)
    return foreground, background, edge, position


class TripleConvBlock(nn.Module):
    """Paper-specified three 3x3 Conv-BN-ReLU layers."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        layers = []
        for index in range(3):
            layers += [nn.Conv2d(in_channels if index == 0 else out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SMUBranch(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        channels = [base_channels * 2**i for i in range(4)]
        self.encoders = nn.ModuleList([TripleConvBlock(in_channels, channels[0])] + [TripleConvBlock(channels[i-1], channels[i]) for i in range(1, 4)])
        self.bridge = TripleConvBlock(channels[-1], channels[-1]*2)
        self.decoders = nn.ModuleList([TripleConvBlock(channels[i]*3, channels[i]) for i in reversed(range(4))])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for block in self.encoders:
            x = block(x); skips.append(x); x = self.pool(x)
        x = self.bridge(x); decoded = []
        for block, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1)); decoded.append(x)
        return decoded


class MorphologyStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(); self.block = TripleConvBlock(input_channels, output_channels)

    def forward(self, tensors: Sequence[torch.Tensor], size: Tuple[int, int]) -> torch.Tensor:
        tensors = [t if t.shape[-2:] == size else F.interpolate(t, size, mode="bilinear", align_corners=False) for t in tensors]
        return self.block(torch.cat(tensors, 1))


class SMUNet(nn.Module):
    """SMU-Net foreground/background U-Nets and four-stage morphology stream."""
    saliency_cluster_counts = (8, 15, 50)

    def __init__(self, in_channels: int = 3, base_channels: int = 64, saliency_backend: str = "tensor", strict_lsc: bool = True, saliency_cache_dir: Optional[str] = None, **_: object):
        super().__init__()
        if base_channels < 4: raise ValueError("base_channels must be at least 4")
        self.foreground_branch, self.background_branch = SMUBranch(in_channels+1, base_channels), SMUBranch(in_channels+1, base_channels)
        branch_channels = [base_channels*8, base_channels*4, base_channels*2, base_channels]
        middle_channels = [base_channels*4, base_channels*2, base_channels, base_channels//2]
        previous = 0; self.middle_stages = nn.ModuleList()
        for index, (branch, output) in enumerate(zip(branch_channels, middle_channels)):
            self.middle_stages.append(MorphologyStage(previous + 2*branch + (index > 0), output)); previous = output
        self.foreground_head, self.background_head = nn.Conv2d(base_channels, 1, 1), nn.Conv2d(base_channels, 1, 1)
        self.middle_head = nn.Conv2d(middle_channels[-1], 1, 1)
        self.shape_head, self.edge_head = nn.Conv2d(middle_channels[1], 1, 1), nn.Conv2d(middle_channels[2], 1, 1)
        if saliency_backend not in {"tensor", "lsc"}: raise ValueError("saliency_backend must be tensor or lsc")
        self.saliency_generator = SMUSaliencyGenerator(strict_lsc, cache_dir=saliency_cache_dir) if saliency_backend == "lsc" else None

    def forward(self, image: torch.Tensor, oracle_mask: Optional[torch.Tensor] = None, seed_points: Optional[torch.Tensor] = None, foreground_saliency: Optional[torch.Tensor] = None, background_saliency: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if seed_points is None:
            if oracle_mask is None: raise ValueError("SMU-Net requires seed_points or oracle_mask")
            seed_points = simulate_three_clicks(oracle_mask)
        generated_fg, generated_bg, edge, position = _point_guidance(image, seed_points)
        if self.saliency_generator is not None:
            generated_fg, generated_bg = self.saliency_generator(image, seed_points)
        foreground_saliency = generated_fg if foreground_saliency is None else foreground_saliency
        background_saliency = generated_bg if background_saliency is None else background_saliency
        foreground = self.foreground_branch(torch.cat([image, foreground_saliency], 1))
        background = self.background_branch(torch.cat([image, background_saliency], 1))
        middle = None; outputs = []; cues = [None, foreground_saliency, edge, position]
        for index, (stage, fg, bg) in enumerate(zip(self.middle_stages, foreground, background)):
            inputs = [fg * (1 - torch.sigmoid(bg)) + fg, bg]
            if middle is not None: inputs.insert(0, middle)
            if cues[index] is not None: inputs.append(cues[index])
            middle = stage(inputs, fg.shape[-2:]); outputs.append(middle)
        size = image.shape[-2:]
        foreground_logits, background_logits = self.foreground_head(foreground[-1]), self.background_head(background[-1])
        middle_logits = self.middle_head(outputs[-1])
        return {"logits": middle_logits, "foreground_logits": foreground_logits, "background_logits": background_logits, "middle_logits": middle_logits, "shape_logits": F.interpolate(self.shape_head(outputs[1]), size, mode="bilinear", align_corners=False), "edge_logits": F.interpolate(self.edge_head(outputs[2]), size, mode="bilinear", align_corners=False), "position_map": position, "seed_points": seed_points}


def _adaptive_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits); fraction = target.flatten(1).mean(1).clamp(1e-3, 1-1e-3)
    positive, negative = (1-fraction)[:, None, None, None], fraction[:, None, None, None]
    intersection = (positive * probability * target).flatten(1).sum(1)
    denominator = (positive*(probability+target) + negative*(2-probability-target)).flatten(1).sum(1)
    return (1 - (2*intersection+1)/(denominator+1)).mean()


def _mask_edge(mask: torch.Tensor) -> torch.Tensor:
    return (F.max_pool2d(mask, 3, 1, 1) + F.max_pool2d(-mask, 3, 1, 1)).clamp(0, 1)


class SMUNetLoss(nn.Module):
    def __init__(self, unet_weight: float = 1, middle_weight: float = 1, shape_weight: float = .5, dice_weight: float = 1, bce_weight: float = 1):
        super().__init__(); self.weights = unet_weight, middle_weight, shape_weight; self.dice_weight, self.bce_weight = dice_weight, bce_weight

    def _term(self, logits, target):
        return self.dice_weight*_adaptive_dice_loss(logits, target) + self.bce_weight*F.binary_cross_entropy_with_logits(logits, target)

    def forward(self, output: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        foreground = self._term(output["foreground_logits"], target); background = self._term(output["background_logits"], 1-target)
        middle = self._term(output["middle_logits"], target)
        shape = F.mse_loss(torch.sigmoid(output["shape_logits"]), target)
        edge = F.binary_cross_entropy_with_logits(output["edge_logits"], _mask_edge(target))
        return self.weights[0]*(foreground+background) + self.weights[1]*middle + self.weights[2]*(shape+edge)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        padding = dilation*(kernel_size//2)
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))


class HybridAdaptiveAttention(nn.Module):
    """HAAM with complementary channel and spatial attention."""
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, output_kernel_size: int = 3):
        super().__init__(); out_channels = out_channels or in_channels
        self.dilated, self.large = ConvBNReLU(in_channels, out_channels, 3, 3), ConvBNReLU(in_channels, out_channels, 5)
        self.channel_mlp = nn.Sequential(nn.Linear(2*out_channels, out_channels), nn.LayerNorm(out_channels), nn.ReLU(True), nn.Linear(out_channels, out_channels), nn.Sigmoid())
        self.channel_fusion = ConvBNReLU(2*out_channels, out_channels, 1)
        self.spatial = nn.Sequential(ConvBNReLU(in_channels, out_channels, 3), ConvBNReLU(out_channels, out_channels, 1))
        self.spatial_attention = nn.Sequential(nn.ReLU(True), nn.Conv2d(out_channels, 1, 1), nn.Sigmoid())
        self.output = ConvBNReLU(2*out_channels, out_channels, output_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilated, large = self.dilated(x), self.large(x)
        weight = self.channel_mlp(torch.cat([dilated, large], 1).mean((2, 3)))[:, :, None, None]
        channel = self.channel_fusion(torch.cat([dilated*weight, large*(1-weight)], 1)); spatial = self.spatial(x)
        spatial_weight = self.spatial_attention(channel+spatial)
        return self.output(torch.cat([channel*spatial_weight, spatial*(1-spatial_weight)], 1))


class AAUStage(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(HybridAdaptiveAttention(in_channels, out_channels), HybridAdaptiveAttention(out_channels, out_channels))


class AAUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, **_: object):
        super().__init__(); channels = [base_channels*2**i for i in range(4)]
        self.encoders = nn.ModuleList([AAUStage(in_channels, channels[0])] + [AAUStage(channels[i-1], channels[i]) for i in range(1, 4)])
        self.bridge = AAUStage(channels[-1], channels[-1]*2)
        self.decoders = nn.ModuleList([AAUStage(channels[i]*3, channels[i]) for i in reversed(range(4))])
        self.pool, self.head = nn.MaxPool2d(2), nn.Conv2d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders: x = encoder(x); skips.append(x); x = self.pool(x)
        x = self.bridge(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(torch.cat([F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.head(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: Sequence[int] = (1, 6, 12, 18)):
        super().__init__(); self.branches = nn.ModuleList([ConvBNReLU(in_channels, out_channels, 1 if r == 1 else 3, r) for r in rates]); self.project = ConvBNReLU(len(rates)*out_channels, out_channels, 1)

    def forward(self, x): return self.project(torch.cat([branch(x) for branch in self.branches], 1))


class TinyBRNBackbone(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__(); self.stem = ConvBNReLU(in_channels, base_channels, 3); self.fine = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base_channels, base_channels*2, 3)); self.coarse = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(base_channels*2, base_channels*4, 3), nn.MaxPool2d(2), ConvBNReLU(base_channels*4, base_channels*8, 3, 2)); self.fine_channels, self.coarse_channels = base_channels*2, base_channels*8

    def forward(self, x): x = self.stem(x); fine = self.fine(x); return fine, self.coarse(fine)


class ResNet101BRNBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__(); from torchvision.models import resnet101
        network = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        if in_channels != 3: network.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool); self.layer1, self.layer2, self.layer3, self.layer4 = network.layer1, network.layer2, network.layer3, network.layer4; self.fine_channels, self.coarse_channels = 256, 2048

    def forward(self, x): x = self.stem(x); fine = self.layer1(x); x = self.layer2(fine); x = self.layer3(x); return fine, self.layer4(x)


def _grid(height, width, device, dtype):
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=device, dtype=dtype), torch.linspace(0, 1, width, device=device, dtype=dtype), indexing="ij")
    return torch.stack([xx, yy], -1)


def _sample(feature, points):
    return F.grid_sample(feature, points.mul(2).sub(1).unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)


class CyclicGraphBlock(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.standard = nn.ModuleList([nn.Linear(channels, channels) for _ in range(2)]); self.residual = nn.ModuleList([nn.Linear(channels, channels) for _ in range(3)]); self.norm = nn.LayerNorm(channels)

    @staticmethod
    def aggregate(x): return (x+x.roll(1,1)+x.roll(-1,1)+x.roll(2,1)+x.roll(-2,1))/5

    def forward(self, nodes):
        x = nodes
        for layer in self.standard: x = F.relu(layer(self.aggregate(x)))
        residual = x
        for layer in self.residual: x = F.relu(layer(self.aggregate(x)))
        return self.norm(x+residual)


class BRN(nn.Module):
    """Binary BRN: multi-scale CNN, boundary sampling and graph refinement."""
    def __init__(self, in_channels: int = 3, base_channels: int = 64, backbone: str = "resnet101", boundary_nodes: int = 60, boundary_candidates: int = 256, evaluation_candidates: int = 4096, uncertainty_ratio: float = .75, graph_train_samples: int = 3, graph_eval_samples: int = 5, graph_blocks: int = 3, max_offset: float = .09, **_: object):
        super().__init__()
        self.boundary_nodes = int(boundary_nodes)
        self.boundary_candidates = int(boundary_candidates)
        self.evaluation_candidates = int(evaluation_candidates)
        self.uncertainty_ratio = float(uncertainty_ratio)
        self.graph_train_samples = int(graph_train_samples)
        self.graph_eval_samples = int(graph_eval_samples)
        self.max_offset = float(max_offset)
        if backbone == "resnet101": self.backbone = ResNet101BRNBackbone(in_channels)
        elif backbone == "tiny": self.backbone = TinyBRNBackbone(in_channels, base_channels)
        else: raise ValueError("BRN backbone must be 'resnet101' or 'tiny'")
        self.aspp = ASPP(self.backbone.coarse_channels, base_channels); self.coarse_head = nn.Conv2d(base_channels, 1, 1)
        point_channels = self.backbone.fine_channels+1
        self.boundary_classifier = nn.Sequential(nn.Linear(point_channels, base_channels), nn.ReLU(True), nn.Linear(base_channels, 1))
        self.contour_projection = nn.Linear(point_channels, base_channels); self.graph_blocks = nn.ModuleList([CyclicGraphBlock(base_channels) for _ in range(graph_blocks)]); self.offset_head = nn.Linear(base_channels, 2); self.render_scale = nn.Parameter(torch.tensor(1.))

    def _boundary_samples(self, fine, coarse):
        uncertainty = 1-torch.abs(2*torch.sigmoid(coarse)-1)
        batch, _, height, width = uncertainty.shape
        requested = self.boundary_candidates if self.training else self.evaluation_candidates
        count = min(requested, height*width)
        uncertain_count = min(count, int(round(self.uncertainty_ratio*count))) if self.training else count
        indices = uncertainty.flatten(1).topk(uncertain_count, 1).indices
        random_count = count-uncertain_count
        if random_count:
            random_scores = torch.rand(batch, height*width, device=coarse.device)
            random_scores.scatter_(1, indices, -1.0)
            random_indices = random_scores.topk(random_count, 1).indices
            indices = torch.cat([indices, random_indices], 1)
        points = torch.stack([(indices%width).to(coarse.dtype)/max(width-1,1), torch.div(indices,width,rounding_mode="floor").to(coarse.dtype)/max(height-1,1)], -1)
        features = torch.cat([_sample(fine, points), _sample(coarse, points)], -1)
        return points, self.boundary_classifier(features).squeeze(-1)

    def _initial_contour(self, probability, boundary_points, boundary_logits):
        batch, _, height, width = probability.shape
        grid = _grid(height, width, probability.device, probability.dtype)
        weights = probability[:,0].clamp_min(1e-6)
        coarse_centre = (weights[...,None]*grid).sum((1,2))/weights.sum((1,2))[:,None]
        coarse_radius = torch.sqrt(weights.mean((1,2)).clamp(1e-4,.8)/math.pi).clamp(.03,.45)
        boundary_weights = torch.sigmoid(boundary_logits).clamp_min(1e-6)
        boundary_centre = (boundary_points*boundary_weights[...,None]).sum(1)/boundary_weights.sum(1,keepdim=True)
        boundary_radius = torch.linalg.vector_norm(boundary_points-boundary_centre[:,None,:],dim=-1)
        boundary_radius = (boundary_radius*boundary_weights).sum(1)/boundary_weights.sum(1)
        centre = (coarse_centre+boundary_centre)/2
        radius = ((coarse_radius+boundary_radius)/2).clamp(.03,.45)
        angles = torch.arange(self.boundary_nodes, device=probability.device, dtype=probability.dtype)*2*math.pi/self.boundary_nodes
        circle = torch.stack([torch.cos(angles),torch.sin(angles)],-1)
        return (centre[:,None,:]+radius[:,None,None]*circle[None]).clamp(0,1)

    def _refine(self, fine, coarse, contour):
        stage_points, stage_logits = [], []
        for block in self.graph_blocks:
            contour, points, logits = self._graph_neighbourhood(fine, coarse, contour)
            stage_points.append(points); stage_logits.append(logits)
            features = torch.cat([_sample(fine, contour), _sample(coarse, contour)], -1)
            nodes = block(self.contour_projection(features))
            contour = (contour+torch.tanh(self.offset_head(nodes))*self.max_offset).clamp(0,1)
        return contour, torch.stack(stage_points, 1), torch.stack(stage_logits, 1)

    def _graph_neighbourhood(self, fine, coarse, contour):
        sample_count = self.graph_train_samples if self.training else self.graph_eval_samples
        centre = contour.mean(1, keepdim=True)
        direction = F.normalize(contour-centre, dim=-1)
        offsets = torch.linspace(-self.max_offset, self.max_offset, sample_count, device=contour.device, dtype=contour.dtype)
        points = (contour[:,:,None,:]+direction[:,:,None,:]*offsets[None,None,:,None]).clamp(0,1)
        flattened = points.flatten(1,2)
        features = torch.cat([_sample(fine, flattened), _sample(coarse, flattened)], -1)
        logits = self.boundary_classifier(features).squeeze(-1).view(contour.shape[0], contour.shape[1], sample_count)
        refined = (points*torch.softmax(logits,2)[...,None]).sum(2)
        return refined, points, logits

    @staticmethod
    def _render(contour, height, width):
        distance = ((_grid(height,width,contour.device,contour.dtype)[None,:,:,None,:]-contour[:,None,None,:,:])**2).sum(-1).amin(-1); sigma=2/max(height,width); return torch.exp(-distance/(2*sigma**2)).unsqueeze(1)

    def forward(self, image):
        size=image.shape[-2:]
        fine, coarse_features=self.backbone(image)
        coarse=F.interpolate(self.coarse_head(self.aspp(coarse_features)),size,mode="bilinear",align_corners=False)
        boundary_points,boundary_logits=self._boundary_samples(fine,coarse)
        contour=self._initial_contour(torch.sigmoid(coarse),boundary_points,boundary_logits)
        contour,graph_points_by_stage,graph_logits_by_stage=self._refine(fine,coarse,contour)
        graph_points=graph_points_by_stage[:,-1]
        graph_point_logits=graph_logits_by_stage[:,-1]
        rendered=self._render(contour,*size)
        return {"logits":coarse+self.render_scale*rendered,"coarse_logits":coarse,"boundary_logits":boundary_logits,"boundary_points":boundary_points,"contour_points":contour,"graph_points":graph_points,"graph_point_logits":graph_point_logits,"graph_points_by_stage":graph_points_by_stage,"graph_logits_by_stage":graph_logits_by_stage,"rendered_boundary":rendered}


def _mask_contour_points(mask, count):
    _,_,height,width=mask.shape; outputs=[]
    for edge in _mask_edge(mask)[:,0]:
        coords=torch.nonzero(edge>.5,as_tuple=False)
        if not coords.numel(): outputs.append(mask.new_full((count,2),.5)); continue
        centre=coords.float().mean(0); angles=torch.atan2(coords[:,0].float()-centre[0],coords[:,1].float()-centre[1]); ordered=coords[angles.argsort()]; selected=ordered[torch.linspace(0,len(ordered)-1,count,device=mask.device).round().long()]; outputs.append(torch.stack([selected[:,1].to(mask.dtype)/max(width-1,1),selected[:,0].to(mask.dtype)/max(height-1,1)],-1))
    return torch.stack(outputs)


class BRNLoss(nn.Module):
    def __init__(self, coarse_weight=1., boundary_weight=1., graph_weight=1., refined_weight=1.): super().__init__(); self.weights=coarse_weight,boundary_weight,graph_weight,refined_weight

    def forward(self, output: Dict[str,torch.Tensor], target: torch.Tensor):
        coarse=F.binary_cross_entropy_with_logits(output["coarse_logits"],target)
        refined=F.binary_cross_entropy_with_logits(output["logits"],target)+_adaptive_dice_loss(output["logits"],target)
        edge=_mask_edge(target)
        edge_target=_sample(edge,output["boundary_points"]).squeeze(-1)
        boundary=F.binary_cross_entropy_with_logits(output["boundary_logits"],edge_target)
        graph_points=output["graph_points_by_stage"].flatten(1,3)
        graph_target=_sample(edge,graph_points).squeeze(-1).view_as(output["graph_logits_by_stage"])
        boundary=boundary+F.binary_cross_entropy_with_logits(output["graph_logits_by_stage"],graph_target)
        target_contour=_mask_contour_points(target,output["contour_points"].shape[1])
        cyclic=torch.stack([((output["contour_points"]-target_contour.roll(shift,1))**2).sum(-1).mean(1) for shift in range(target_contour.shape[1])],1).amin(1).mean()
        return self.weights[0]*coarse+self.weights[1]*boundary+self.weights[2]*cyclic+self.weights[3]*refined


class MedSAMStandardDecoder(nn.Module):
    def __init__(self, checkpoint_path=None, base_channels=64, freeze_medsam=True, **_: object):
        super().__init__(); from bua_lel.models.backbones.medsam import MedSAMMultiScaleEncoder
        self.encoder=MedSAMMultiScaleEncoder(checkpoint_path=checkpoint_path,freeze_medsam=freeze_medsam,return_cls_feat=False); self.decoder=nn.Sequential(nn.Conv2d(256,base_channels*2,3,padding=1),nn.ReLU(True),nn.ConvTranspose2d(base_channels*2,base_channels,4,2,1),nn.ReLU(True),nn.ConvTranspose2d(base_channels,base_channels//2,4,2,1),nn.ReLU(True),nn.Conv2d(base_channels//2,1,1))

    def forward(self,x): size=x.shape[-2:]; embedding=self.encoder(x,return_dict=True)["medsam_neck"]; return F.interpolate(self.decoder(embedding),size=size,mode="bilinear",align_corners=False)
