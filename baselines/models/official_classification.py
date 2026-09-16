"""Official classification."""
from __future__ import annotations

import math
import hashlib
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TinyMultilevelImageEncoder(nn.Module):
    output_dims = (32, 64, 128)

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU()),
            nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU()),
        ])

    def forward(self, image):
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image.mean((2, 3)))
        return outputs


class MultiLevelViTL32Encoder(nn.Module):
    """Frozen ImageNet ViT-L/32 features at author-source layers 6, 9 and 11."""
    output_dims = (1024, 1024, 1024)

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ViT_L_32_Weights, vit_l_32
        self.network = vit_l_32(weights=ViT_L_32_Weights.DEFAULT if pretrained else None)
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        self.finetune_blocks = 0
        self.finetune_active = False
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self._feature_cache = {}

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()
        if self.finetune_blocks and self.finetune_active:
            for layer in self.network.encoder.layers[-self.finetune_blocks:]:
                layer.train(mode)
            self.network.encoder.ln.train(mode)
        return self

    def enable_finetune(self, blocks=3, active=True):
        blocks = int(blocks)
        if not 1 <= blocks <= len(self.network.encoder.layers):
            raise ValueError("finetune blocks must be within the ViT encoder depth")
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        for layer in self.network.encoder.layers[-blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.network.encoder.ln.parameters():
            parameter.requires_grad = True
        self.finetune_blocks = blocks
        self.finetune_active = bool(active)
        self._feature_cache.clear()
        return self

    def set_finetune_active(self, active=True):
        if not self.finetune_blocks:
            raise RuntimeError("enable_finetune must be called before activation")
        self.finetune_active = bool(active)
        self._feature_cache.clear()
        return self

    @staticmethod
    def _cache_key(image):
        thumbnail = F.adaptive_avg_pool2d(image.detach(), (16, 16)).to("cpu", torch.float16).numpy()
        return hashlib.sha1(thumbnail.tobytes()).hexdigest()

    def _encode(self, image):
        image = F.interpolate((image-self.mean)/self.std, (224, 224), mode="bilinear", align_corners=False)
        context = torch.enable_grad() if self.finetune_blocks and self.finetune_active else torch.no_grad()
        with context:
            tokens = self.network._process_input(image)
            tokens = torch.cat([self.network.class_token.expand(image.shape[0], -1, -1), tokens], dim=1)
            tokens = self.network.encoder.dropout(tokens + self.network.encoder.pos_embedding)
            outputs = []
            for index, layer in enumerate(self.network.encoder.layers):
                tokens = layer(tokens)
                if index in {6, 9, 11}:
                    outputs.append(tokens.mean(1))
            return outputs

    def forward(self, image):
        if self.finetune_blocks and self.finetune_active:
            return self._encode(image)
        keys = [self._cache_key(sample.unsqueeze(0)) for sample in image]
        missing_indices = [index for index, key in enumerate(keys) if key not in self._feature_cache]
        if missing_indices:
            encoded = self._encode(image[missing_indices])
            for local_index, batch_index in enumerate(missing_indices):
                self._feature_cache[keys[batch_index]] = tuple(level[local_index].detach().float().cpu() for level in encoded)
        return [torch.stack([self._feature_cache[key][level] for key in keys]).to(image.device) for level in range(3)]


def _normalized_adjacency(nodes: int, edges: Iterable[tuple[int, int]] | None = None):
    adjacency = torch.eye(nodes)
    if edges is None:
        adjacency.fill_(1.0)
    else:
        for source, target in edges:
            if source < nodes and target < nodes:
                adjacency[source, target] = adjacency[target, source] = 1.0
    degree = adjacency.sum(1).clamp_min(1.0)
    return adjacency / torch.sqrt(degree[:, None] * degree[None, :])


class ClinicalGraphEncoder(nn.Module):
    def __init__(self, clinical_dim, hidden_dim=64, layers=3, edges=None, feature_slices=None):
        super().__init__()
        self.feature_slices = tuple(tuple(map(int, item)) for item in (feature_slices or [(i, i+1) for i in range(clinical_dim)]))
        node_count = len(self.feature_slices)
        self.node_scale = nn.Parameter(torch.empty(node_count, hidden_dim))
        self.node_bias = nn.Parameter(torch.zeros(node_count, hidden_dim))
        nn.init.xavier_uniform_(self.node_scale)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)])
        self.register_buffer("adjacency", _normalized_adjacency(node_count, edges), persistent=True)

    @property
    def edge_index(self):
        return self.adjacency.nonzero().t()

    def forward(self, clinical):
        values = torch.stack([clinical[:, start:end].mean(1) for start, end in self.feature_slices], 1)
        nodes = values.unsqueeze(-1)*self.node_scale.unsqueeze(0) + self.node_bias.unsqueeze(0)
        outputs = []
        for layer in self.layers:
            nodes = F.gelu(layer(torch.einsum("ij,bjh->bih", self.adjacency, nodes)))
            outputs.append(nodes.mean(1))
        return outputs


class InterconnectionBlock(nn.Module):
    def __init__(self, image_dim, graph_dim, clinical_dim, output_dim=64):
        super().__init__()
        self.image = nn.Linear(image_dim, output_dim)
        self.graph = nn.Linear(graph_dim, output_dim)
        self.clinical = nn.Linear(clinical_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, image, graph, clinical):
        tokens = torch.stack([self.image(image), self.graph(graph), self.clinical(clinical)], 1)
        attended, weights = self.attention(tokens, tokens, tokens, need_weights=True)
        return self.norm(tokens + attended).mean(1), weights


class MIINetClassifier(nn.Module):
    """Multi-level image/clinical-graph interconnection with channel shuffle."""
    def __init__(self, clinical_dim, num_classes=3, pretrained=True, lightweight=False, hidden_dim=64, clinical_feature_slices=None, **_):
        super().__init__()
        self.image_encoder = _TinyMultilevelImageEncoder() if lightweight else MultiLevelViTL32Encoder(pretrained)
        self.checkpoint_exclude_prefixes = () if lightweight else ("image_encoder.network.",)
        self.clinical_graph_encoder = ClinicalGraphEncoder(clinical_dim, hidden_dim, layers=3, feature_slices=clinical_feature_slices)
        self.interconnection_blocks = nn.ModuleList([
            InterconnectionBlock(image_dim, hidden_dim, clinical_dim, hidden_dim)
            for image_dim in self.image_encoder.output_dims
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*3+clinical_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.5), nn.Linear(256, num_classes),
        )

    @staticmethod
    def _channel_shuffle(features, groups=5):
        while groups > 1 and features.shape[1] % groups:
            groups -= 1
        if groups == 1:
            return features
        return features.view(features.shape[0], groups, -1).transpose(1, 2).reshape_as(features)

    def forward(self, image, clinical):
        image_levels = self.image_encoder(image)
        graph_levels = self.clinical_graph_encoder(clinical)
        fused, weights = [], []
        for block, image_level, graph_level in zip(self.interconnection_blocks, image_levels, graph_levels):
            level, attention = block(image_level, graph_level, clinical)
            fused.append(level); weights.append(attention)
        representation = self._channel_shuffle(torch.cat([*fused, clinical], 1))
        return {"logits": self.classifier(representation), "interconnection_attention": weights}


class CNNViTImageEncoder(nn.Module):
    """CVUIF: local DenseNet features followed by global transformer mixing."""
    def __init__(self, pretrained=True, lightweight=False, output_dim=192):
        super().__init__()
        if lightweight:
            self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU())
            channels = 64
        else:
            from torchvision.models import DenseNet121_Weights, densenet121
            dense = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None).features
            self.cnn = nn.Sequential(*list(dense.children())[:6])
            channels = 128
        self.projection = nn.Conv2d(channels, output_dim, 1)
        layer = nn.TransformerEncoderLayer(output_dim, 6, output_dim*4, .1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 4 if not lightweight else 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.output_dim = output_dim
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, image):
        feature = self.projection(self.cnn(image))
        feature = F.adaptive_avg_pool2d(feature, (14, 14) if feature.shape[-1] >= 14 else (8, 8))
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(image.shape[0], -1, -1), tokens], 1)
        return self.transformer(tokens)[:, 0]


HER2_NODE_NAMES = ("Age", "\u4e73\u623f\u624b\u672f", "\u814b\u7a9d\u624b\u672f", "\u75c5\u7406\u5b66\u7c7b\u578b", "size", "diff", "\u4e73\u5934\u6216\u76ae\u80a4\u53d7\u7d2f", "LVI", "ER", "PR", "Ki-67", "EGFR")
HER2_CAUSAL_EDGES = ((3, 10), (4, 10), (5, 10), (10, 7), (7, 1), (7, 2), (6, 1), (6, 2), (4, 1), (4, 2))


class ReducedDimensionalFusion(nn.Module):
    def __init__(self, image_dim, graph_dim, clinical_dim, reduced=50, fused=128):
        super().__init__()
        self.image = nn.Linear(image_dim, reduced)
        self.graph = nn.Linear(graph_dim, reduced)
        self.clinical = nn.Linear(clinical_dim, reduced)
        self.gate = nn.Sequential(nn.Linear(reduced*2, fused), nn.GELU(), nn.Linear(fused, reduced), nn.Sigmoid())
        self.output_dim = reduced*2

    def forward(self, image, graph, clinical):
        image, graph, clinical = self.image(image), self.graph(graph), self.clinical(clinical)
        gate = self.gate(torch.cat([image, graph], 1))
        return torch.cat([gate*image+(1-gate)*graph, clinical], 1)


class KMNetClassifier(nn.Module):
    """Knowledge-guided clinical GCN + CNN/ViT ultrasound encoder + RDF."""
    def __init__(self, clinical_dim, num_classes=3, pretrained=True, lightweight=False, clinical_feature_names=None, clinical_feature_slices=None, **_):
        super().__init__()
        self.image_encoder = CNNViTImageEncoder(pretrained, lightweight)
        node_count = len(clinical_feature_slices) if clinical_feature_slices else clinical_dim
        edges = HER2_CAUSAL_EDGES if node_count == 12 and not clinical_feature_names else self._semantic_edges(clinical_feature_names or [])
        self.knowledge_graph_encoder = ClinicalGraphEncoder(clinical_dim, 64, layers=3, edges=edges, feature_slices=clinical_feature_slices)
        self.rdf = ReducedDimensionalFusion(self.image_encoder.output_dim, 64, clinical_dim)
        self.classifier = nn.Sequential(nn.Linear(self.rdf.output_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(.5), nn.Linear(128, num_classes))

    @staticmethod
    def _semantic_edges(names: Sequence[str]):
        normalized = [str(name).lower().replace("（cm）", "").replace("%", "") for name in names]
        aliases = {"size": ("size", "\u5927\u5c0f", "\u6781\u5f84"), "diff": ("diff", "\u5206\u5316"), "skin": ("\u4e73\u5934", "\u76ae\u80a4"), "lvi": ("lvi",), "ki67": ("ki-67", "ki67"), "breast": ("\u4e73\u623f\u624b\u672f",), "axilla": ("\u814b\u7a9d\u624b\u672f",), "pathology": ("\u75c5\u7406",)}
        found = {key: next((i for i, name in enumerate(normalized) if any(alias in name for alias in values)), None) for key, values in aliases.items()}
        semantic = (("pathology", "ki67"), ("size", "ki67"), ("diff", "ki67"), ("ki67", "lvi"), ("lvi", "breast"), ("lvi", "axilla"), ("skin", "breast"), ("skin", "axilla"), ("size", "breast"), ("size", "axilla"))
        return [(found[a], found[b]) for a, b in semantic if found[a] is not None and found[b] is not None]

    def forward(self, image, clinical):
        image_feature = self.image_encoder(image)
        graph_feature = self.knowledge_graph_encoder(clinical)[-1]
        fused = self.rdf(image_feature, graph_feature, clinical)
        return self.classifier(fused)


class ClinicalEmbedding(nn.Module):
    def __init__(self, clinical_dim, output_dim=8):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(clinical_dim, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, clinical):
        return self.network(clinical)


class HyperConv2d(nn.Module):
    """Per-patient convolution whose parameters are generated from tabular data."""
    def __init__(self, in_channels, out_channels, embedding_dim, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight_generator = nn.Linear(embedding_dim, out_channels*in_channels*kernel_size*kernel_size)
        self.bias_generator = nn.Linear(embedding_dim, out_channels)
        nn.init.zeros_(self.weight_generator.bias); nn.init.zeros_(self.bias_generator.bias)

    def initialize_input_variance(self, clinical: torch.Tensor) -> None:
        variances = clinical.detach().float().var(dim=1, unbiased=False)
        input_variance = float(variances.mean().clamp_min(1e-12))
        weight_bound = math.sqrt(3.0 / (self.in_channels * self.weight_generator.in_features * input_variance))
        bias_bound = math.sqrt(3.0 / (self.bias_generator.in_features * input_variance))
        nn.init.uniform_(self.weight_generator.weight, -weight_bound, weight_bound)
        nn.init.uniform_(self.bias_generator.weight, -bias_bound, bias_bound)
        nn.init.zeros_(self.weight_generator.bias); nn.init.zeros_(self.bias_generator.bias)

    def forward(self, image, embedding):
        batch, _, height, width = image.shape
        weights = self.weight_generator(embedding).view(batch*self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        bias = self.bias_generator(embedding).reshape(-1)
        grouped = F.conv2d(image.reshape(1, batch*self.in_channels, height, width), weights, bias, self.stride, self.padding, groups=batch)
        return grouped.view(batch, self.out_channels, grouped.shape[-2], grouped.shape[-1])


class HyperResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, stride=2, dropout=.3, bn_momentum=.05):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(in_channels, momentum=bn_momentum), nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.downsample = HyperConv2d(in_channels, out_channels, embedding_dim, 1, stride)
        self.downsample_bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, image, embedding):
        identity = self.downsample_bn(self.downsample(image, embedding))
        output = self.conv1(self.dropout(F.relu(self.bn1(image))))
        output = self.conv2(self.dropout(F.relu(self.bn2(output))))
        return output + identity


class _PreactiveResidual2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=.1, bn_momentum=.05):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride), nn.BatchNorm2d(out_channels, momentum=bn_momentum))
    def forward(self, image):
        identity = self.skip(image)
        output = self.conv1(self.dropout(F.relu(self.bn1(image))))
        output = self.conv2(self.dropout(F.relu(self.bn2(output))))
        return output + identity


class HyperFusionClassifier(nn.Module):
    """2-D ultrasound adaptation of the authors' tabular-conditioned hypernetwork."""
    def __init__(self, clinical_dim, num_classes=3, lightweight=False, **_):
        super().__init__()
        channels = (4, 8, 16, 32, 64)
        self.clinical_embedding = ClinicalEmbedding(clinical_dim, 8)
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0], momentum=.05), nn.ReLU(), nn.MaxPool2d(2))
        self.blocks = nn.Sequential(
            _PreactiveResidual2D(channels[0], channels[1], stride=1, dropout=.1),
            _PreactiveResidual2D(channels[1], channels[2], stride=2, dropout=.2),
            _PreactiveResidual2D(channels[2], channels[3], stride=2, dropout=.2),
        )
        self.hyper_residual_block = HyperResidualBlock2D(channels[3], channels[4], 8, stride=2, dropout=.3)
        self.classifier = nn.Sequential(nn.Dropout(.6), nn.Linear(channels[4], channels[2]), nn.ReLU(), nn.Dropout(.5), nn.Linear(channels[2], num_classes))

    def forward(self, image, clinical):
        embedding = self.clinical_embedding(clinical)
        feature = self.hyper_residual_block(self.blocks(self.stem(image)), embedding).mean((2, 3))
        return self.classifier(feature)


class SelectiveStateSpaceBlock(nn.Module):
    """Dependency-free selective sequence mixer used for the FT-Mamba port."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.input = nn.Linear(dim, dim*2)
        self.depthwise = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.output = nn.Linear(dim, dim)
    def forward(self, tokens):
        value, gate = self.input(self.norm(tokens)).chunk(2, -1)
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        return tokens + self.output(F.silu(value)*torch.sigmoid(gate))


class FTMambaEncoder(nn.Module):
    """Official two-layer Mamba tabular encoder with sequence-average pooling."""
    def __init__(self, clinical_dim, dim=64, lightweight=False):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(clinical_dim, dim)); self.bias = nn.Parameter(torch.zeros(clinical_dim, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.xavier_uniform_(self.scale)
        if lightweight:
            self.layers = nn.Sequential(SelectiveStateSpaceBlock(dim), SelectiveStateSpaceBlock(dim))
            self.uses_official_mamba_operator = False
        else:
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise RuntimeError("formal AMF-MedIT requires mamba-ssm; the dependency-free mixer is CPU-test-only") from exc
            self.layers = nn.Sequential(Mamba(d_model=dim, d_state=16, d_conv=4, expand=2), Mamba(d_model=dim, d_state=16, d_conv=4, expand=2))
            self.uses_official_mamba_operator = True


        self.pooling_method = "avg"
        self.norm = nn.LayerNorm(dim); self.output_dim = dim
    def forward(self, clinical):
        tokens = clinical.unsqueeze(-1)*self.scale.unsqueeze(0)+self.bias.unsqueeze(0)
        tokens = torch.cat([self.cls.expand(clinical.shape[0], -1, -1), tokens], 1)
        tokens = self.layers(tokens)
        return self.norm(tokens.mean(dim=1))


class AMFFusion(nn.Module):
    def __init__(self, image_dim, tabular_dim, output_dim=128, confidence=1.0):
        super().__init__()
        image_length = int(round(output_dim*confidence/(1+confidence)))
        image_length = min(max(image_length, 1), output_dim-1)
        image_mask = torch.zeros(1, output_dim); image_mask[:, :image_length] = 1
        self.register_buffer("image_mask", image_mask); self.register_buffer("tabular_mask", 1-image_mask)
        self.image_projector = nn.Sequential(nn.Linear(image_dim, image_dim), nn.ReLU(), nn.Linear(image_dim, output_dim))
        self.tabular_projector = nn.Sequential(nn.Linear(tabular_dim, tabular_dim), nn.ReLU(), nn.Linear(tabular_dim, output_dim))
        self.image_length, self.tabular_length = image_length, output_dim-image_length
    def forward(self, image, tabular):
        image, tabular = self.image_projector(image), self.tabular_projector(tabular)
        fused = image*self.image_mask+tabular*self.tabular_mask
        density = ((image*self.image_mask).abs().sum(1)/self.image_length-(tabular*self.tabular_mask).abs().sum(1)/self.tabular_length).abs().mean()
        leakage = .5*((image*self.tabular_mask).abs().sum(1)/self.tabular_length+(tabular*self.image_mask).abs().sum(1)/self.image_length).mean()
        return fused, density, leakage


class _TinyImageVector(nn.Module):
    output_dim = 128
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 128, 3, 2, 1), nn.ReLU())
    def forward(self, image): return self.net(image).mean((2, 3))


class _ResNet50Vector(nn.Module):
    output_dim = 2048
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        network = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None); network.fc = nn.Identity(); self.network = network
        self.register_buffer("mean", torch.tensor([.485,.456,.406]).view(1,3,1,1), persistent=False); self.register_buffer("std", torch.tensor([.229,.224,.225]).view(1,3,1,1), persistent=False)
    def forward(self, image): return self.network((image-self.mean)/self.std)


class AMFMedITClassifier(nn.Module):
    """ResNet-50 + FT selective encoder + mask-based adaptive fusion."""
    def __init__(self, clinical_dim, num_classes=3, pretrained=True, lightweight=False, alignment_weight=.05, confidence=1.0, **_):
        super().__init__()
        self.image_encoder = _TinyImageVector() if lightweight else _ResNet50Vector(pretrained)
        self.ft_mamba = FTMambaEncoder(clinical_dim, 64, lightweight=lightweight)
        self.fusion = AMFFusion(self.image_encoder.output_dim, self.ft_mamba.output_dim, 128, confidence)
        self.classifier = nn.Linear(128, num_classes); self.alignment_weight = float(alignment_weight)
    def forward(self, image, clinical):
        fused, density, leakage = self.fusion(self.image_encoder(image), self.ft_mamba(clinical))
        return {"logits": self.classifier(fused), "density_loss": density, "leakage_loss": leakage, "aux_loss": self.alignment_weight*5*(density+leakage)}


class AxialTransformerBlock(nn.Module):
    def __init__(self, channels, heads):
        super().__init__(); self.row = nn.MultiheadAttention(channels, heads, batch_first=True); self.column = nn.MultiheadAttention(channels, heads, batch_first=True); self.norm = nn.LayerNorm(channels)
    def forward(self, feature):
        batch, channels, height, width = feature.shape
        rows = feature.permute(0,2,3,1).reshape(batch*height,width,channels); rows = self.norm(rows+self.row(rows,rows,rows,need_weights=False)[0])
        grid = rows.view(batch,height,width,channels); columns = grid.permute(0,2,1,3).reshape(batch*width,height,channels); columns = self.norm(columns+self.column(columns,columns,columns,need_weights=False)[0])
        return columns.view(batch,width,height,channels).permute(0,3,2,1)


class HoverTrans2DEncoder(nn.Module):
    """Independent hierarchical row/column-attention port of MAP's US branch."""
    def __init__(self, lightweight=False):
        super().__init__(); dims = (16,32,64) if lightweight else (32,64,128,256)
        stages=[]; incoming=3
        for index, dim in enumerate(dims):
            stages.append(nn.ModuleDict({"patch": nn.Conv2d(incoming,dim,3,2,1), "attention": AxialTransformerBlock(dim, max(1,dim//32))})); incoming=dim
        self.stages=nn.ModuleList(stages); self.output_dim=dims[-1]
    def forward(self,image):
        for stage in self.stages: image=stage["attention"](F.gelu(stage["patch"](image)))
        return image.mean((2,3))


class MAPClassifier(nn.Module):
    """Available-modality MAP adapter: HoverTrans ultrasound plus clinical data."""
    def __init__(self, clinical_dim, num_classes=3, lightweight=False, **_):
        super().__init__(); self.ultrasound_encoder=HoverTrans2DEncoder(lightweight); self.clinical_projection=nn.Linear(clinical_dim,64)
        self.classifier=nn.Sequential(nn.Linear(self.ultrasound_encoder.output_dim+64,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,num_classes))
    def forward(self,image,clinical): return self.classifier(torch.cat([self.ultrasound_encoder(image),self.clinical_projection(clinical)],1))
