"""Classification baselines and leakage-safe inductive HetMed utilities."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class MedSAMStandardFusion(nn.Module):
    """MedSAM global image feature plus clinical MLP and concatenation fusion."""
    def __init__(self, clinical_dim, num_classes=3, checkpoint_path=None, hidden_dim=256, freeze_medsam=True, **_):
        super().__init__()
        from bua_lel.models.backbones.medsam import MedSAMMultiScaleEncoder
        from bua_lel.models.heads import build_medsam_fusion_projections
        self.encoder = MedSAMMultiScaleEncoder(checkpoint_path=checkpoint_path, freeze_medsam=freeze_medsam, return_cls_feat=False)
        self.image_projection, self.clinical_projection = build_medsam_fusion_projections(clinical_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(True), nn.Dropout(.3), nn.Linear(hidden_dim, num_classes))

    def forward(self, image, clinical):
        embedding = self.encoder(image, return_dict=True)["medsam_neck"].mean((2, 3))
        return self.classifier(torch.cat([self.image_projection(embedding), self.clinical_projection(clinical)], 1))


class HetMedImageEncoder(nn.Module):
    """ResNet image representation used before constructing the HetMed graph.

    The author source supports ResNet-18/50 SimCLR encoders. We default to an
    ImageNet ResNet-50 and apply its fixed normalization inside the encoder,
    while the shared dataset protocol itself stays at ``[0, 1]``.
    """
    def __init__(self, backbone: str = "resnet50", output_dim: Optional[int] = None, checkpoint_path: Optional[str] = None, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
        constructors = {"resnet18": (resnet18, ResNet18_Weights.DEFAULT), "resnet50": (resnet50, ResNet50_Weights.DEFAULT)}
        if backbone not in constructors:
            raise ValueError("HetMed image backbone must be resnet18 or resnet50")
        constructor, default_weights = constructors[backbone]
        network = constructor(weights=default_weights if pretrained and checkpoint_path is None else None)
        feature_dim = network.fc.in_features
        network.fc = nn.Identity()
        self.network = network
        output_dim = feature_dim if output_dim is None else int(output_dim)
        self.projection = nn.Identity() if output_dim == feature_dim else nn.Linear(feature_dim, output_dim)
        self.register_buffer("input_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = state.get("model_state", state.get("state_dict", state))
            missing, unexpected = self.load_state_dict(state, strict=False)
            if len(missing) == len(self.state_dict()) or unexpected:
                raise ValueError(f"incompatible HetMed image checkpoint: missing={len(missing)}, unexpected={unexpected}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.network((image - self.input_mean) / self.input_std))


class InductiveHetMedGraphBuilder:
    """Fit multiplex clinical relations on one training fold only.

    Clinical columns are clustered across training patients, matching HetMed's
    column-wise KMeans grouping. Query rows are only connected to the immutable
    training reference bank; query-query edges are never constructed.
    """
    threshold_candidates = (.01, .75, .9)

    def __init__(self, num_relations: int = 4, threshold=None, seed: int = 42, self_connection: float = 3.0):
        self.num_relations, self.threshold, self.seed = int(num_relations), threshold, int(seed)
        self.self_connection = float(self_connection)
        if self.self_connection <= 0:
            raise ValueError("self_connection must be positive")
        self.feature_groups_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.fitted_threshold_ = None

    def fit(self, train_clinical: np.ndarray) -> "InductiveHetMedGraphBuilder":
        train = self._matrix(train_clinical)
        if train.shape[1] < self.num_relations:
            raise ValueError("clinical feature count must be at least num_relations")
        self.mean_ = train.mean(0)
        self.scale_ = train.std(0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (train-self.mean_)/self.scale_

        self.feature_groups_ = KMeans(n_clusters=self.num_relations, random_state=self.seed, n_init=20).fit_predict(standardized.T)
        if self.threshold is None:
            thresholds = self._fit_thresholds(standardized)
        elif np.isscalar(self.threshold):
            thresholds = np.repeat(float(self.threshold), self.num_relations)
        else:
            thresholds = np.asarray(tuple(self.threshold), dtype=np.float32)
            if thresholds.shape != (self.num_relations,):
                raise ValueError("HetMed needs one graph threshold per relation")
        if not np.isfinite(thresholds).all() or ((thresholds < -1.0) | (thresholds > 1.0)).any():
            raise ValueError("cosine thresholds must lie in [-1, 1]")
        self.fitted_threshold_ = tuple(map(float, thresholds))
        return self

    def _fit_thresholds(self, standardized: np.ndarray) -> np.ndarray:
        """Fallback only: fit each relation independently on training nodes.

        Formal runs pass validation-selected thresholds explicitly.  This
        leakage-safe fallback exists for synthetic tests and external callers.
        """
        fitted = []
        for relation in range(self.num_relations):
            columns = np.flatnonzero(self.feature_groups_ == relation)
            similarity = self._cosine(standardized[:, columns], standardized[:, columns])
            np.fill_diagonal(similarity, -np.inf)
            selected = min(self.threshold_candidates)
            for candidate in sorted(self.threshold_candidates, reverse=True):
                if float((similarity >= candidate).any(1).mean()) >= .8:
                    selected = candidate
                    break
            fitted.append(selected)
        return np.asarray(fitted, dtype=np.float32)

    @staticmethod
    def _matrix(values) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("clinical data must be a finite non-empty [N,D] matrix")
        return matrix

    def _check_fitted(self):
        if self.feature_groups_ is None or self.mean_ is None or self.scale_ is None or self.fitted_threshold_ is None:
            raise RuntimeError("fit must be called on the training fold before graph construction")

    def state_dict(self) -> Dict[str, np.ndarray]:
        self._check_fitted()
        thresholds = np.asarray(self.fitted_threshold_, dtype=np.float32)
        threshold_state = thresholds[0] if np.allclose(thresholds, thresholds[0]) else thresholds
        return {
            "feature_groups": self.feature_groups_.copy(), "mean": self.mean_.copy(),
            "scale": self.scale_.copy(), "threshold": np.asarray(threshold_state, dtype=np.float32),
            "self_connection": np.asarray(self.self_connection, dtype=np.float32),
        }

    def _standardize(self, values) -> np.ndarray:
        self._check_fitted(); matrix = self._matrix(values)
        if matrix.shape[1] != len(self.mean_): raise ValueError("clinical feature dimension differs from fitted training fold")
        return (matrix-self.mean_)/self.scale_

    @staticmethod
    def _cosine(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(1e-8)
        reference_norm = np.linalg.norm(reference, axis=1, keepdims=True).clip(1e-8)
        return (query/query_norm) @ (reference/reference_norm).T

    def _graphs(self, query, reference, training: bool) -> torch.Tensor:
        query_scaled, reference_scaled = self._standardize(query), self._standardize(reference)
        relations = []
        for relation in range(self.num_relations):
            columns = np.flatnonzero(self.feature_groups_ == relation)
            similarity = self._cosine(query_scaled[:, columns], reference_scaled[:, columns])
            adjacency = (similarity > self.fitted_threshold_[relation]).astype(np.float32)
            if training:
                if adjacency.shape[0] != adjacency.shape[1]:
                    raise ValueError("training graph must be square")
                adjacency = np.maximum(adjacency, adjacency.T)
                np.fill_diagonal(adjacency, 0.0)
                adjacency += np.eye(adjacency.shape[0], dtype=np.float32) * self.self_connection
                degree = adjacency.sum(1).clip(1e-8)
                adjacency = adjacency / np.sqrt(degree[:, None] * degree[None, :])
            else:
                empty_rows = adjacency.sum(1) == 0
                if empty_rows.any():
                    adjacency[empty_rows, similarity[empty_rows].argmax(1)] = 1.0


                degree = adjacency.sum(1, keepdims=True) + self.self_connection
                adjacency /= degree
            relations.append(torch.from_numpy(adjacency))
        return torch.stack(relations)

    def training_graph(self, train_clinical) -> torch.Tensor:
        return self._graphs(train_clinical, train_clinical, training=True)

    def query_graph(self, query_clinical, train_clinical) -> torch.Tensor:
        return self._graphs(query_clinical, train_clinical, training=False)


class RelationGCN(nn.Module):
    """The author's single linear GCN followed by ReLU."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = .5):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dropout = float(dropout)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, adjacency: torch.Tensor, reference: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        reference = self.projection(F.dropout(reference, self.dropout, training=self.training))
        if adjacency.shape[0] == adjacency.shape[1] and query.shape[0] == reference.shape[0]:
            aggregated = adjacency @ reference
        else:
            query = self.projection(F.dropout(query, self.dropout, training=self.training))
            self_weight = (1.0 - adjacency.sum(1, keepdim=True)).clamp_min(0.0)
            aggregated = adjacency @ reference + self_weight * query
        return F.relu(aggregated)


class HetMedClassifier(nn.Module):
    """Leakage-safe inductive adaptation of the authors' DMGI HetMed.

    The upstream model learns one free consensus vector per transductive node.
    OOF patients cannot safely use such parameters, so this port replaces that
    lookup table with a shared consensus projection.  The relation GCN,
    corruption objective, attention and signed consensus regularizer follow
    the author implementation.
    """
    def __init__(self, image_dim, clinical_dim, num_classes=3, hidden_dim=64, num_relations=4, dropout=.5, **_):
        super().__init__(); self.num_relations = int(num_relations)
        input_dim = int(image_dim) + int(clinical_dim)
        self.relation_gcns = nn.ModuleList([RelationGCN(input_dim, hidden_dim, dropout) for _ in range(self.num_relations)])
        self.relation_attention = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.num_relations)])
        self.consensus_projection = nn.Linear(hidden_dim, hidden_dim)
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _encode_relations(self, adjacency, reference, query):
        return torch.stack([gcn(adjacency[index], reference, query) for index, gcn in enumerate(self.relation_gcns)], 1)

    @staticmethod
    def _row_normalize(features: torch.Tensor) -> torch.Tensor:
        denominator = features.sum(1, keepdim=True)
        fallback = features.abs().sum(1, keepdim=True).clamp_min(1e-8)
        denominator = torch.where(denominator.abs() < 1e-8, fallback, denominator)
        return features / denominator

    @classmethod
    def _node_features(cls, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:


        image = F.normalize(image, p=2, dim=1)
        clinical = F.normalize(clinical, p=2, dim=1) if clinical.shape[1] else clinical
        return cls._row_normalize(torch.cat([image, clinical], 1))

    def _attend(self, relation_embeddings: torch.Tensor):
        scores = torch.cat([layer(relation_embeddings[:, index]) for index, layer in enumerate(self.relation_attention)], 1)
        attention = torch.softmax(scores, 1)
        return (relation_embeddings * attention.unsqueeze(-1)).sum(1), attention

    def forward(self, query_image, query_clinical, reference_image, reference_clinical, relation_graph=None):
        if relation_graph is None:
            raise ValueError("HetMed requires fold-fitted relation_graph; build it with InductiveHetMedGraphBuilder")
        if relation_graph.ndim != 3 or relation_graph.shape[0] != self.num_relations:
            raise ValueError("relation_graph must have shape [R,N_query,N_train]")
        relation_graph = relation_graph.to(device=query_image.device, dtype=query_image.dtype)
        query = self._node_features(query_image, query_clinical)
        reference = self._node_features(reference_image, reference_clinical)
        relation_embeddings = self._encode_relations(relation_graph, reference, query)
        positive_fusion, attention = self._attend(relation_embeddings)
        consensus = self.consensus_projection(positive_fusion)
        logits = self.classifier(consensus)

        dmgi_loss = logits.sum()*0.0
        positive_distance = ((consensus-positive_fusion)**2).sum()
        negative_distance = positive_distance.detach()*0.0
        if relation_graph.shape[1] == relation_graph.shape[2] and query_image.shape[0] == reference_image.shape[0]:
            permutation = torch.randperm(reference.shape[0], device=reference.device)
            corrupted = reference[permutation]
            negative_embeddings = self._encode_relations(relation_graph, corrupted, corrupted)
            negative_fusion, _ = self._attend(negative_embeddings)
            negative_distance = ((consensus-negative_fusion)**2).sum()
            terms = []
            for relation in range(self.num_relations):
                context = torch.sigmoid(relation_embeddings[:, relation].mean(0, keepdim=True)).expand_as(relation_embeddings[:, relation])
                positive = self.discriminator(relation_embeddings[:, relation], context).squeeze(-1)
                negative = self.discriminator(negative_embeddings[:, relation], context).squeeze(-1)
                terms.append(F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
            dmgi_loss = torch.stack(terms).sum()
        consensus_loss = positive_distance-negative_distance
        return {
            "logits": logits, "relation_embeddings": relation_embeddings,
            "relation_attention": attention, "consensus": consensus,
            "dmgi_loss": dmgi_loss, "consensus_loss": consensus_loss,
            "positive_consensus_distance": positive_distance,
            "negative_consensus_distance": negative_distance,
        }


class HetMedLoss(nn.Module):
    """DMGI mutual information + consensus regularizer + supervised CE."""
    def __init__(self, supervised_coefficient: float = .01, regularization_coefficient: float = .001, class_weight=None, label_smoothing: float = .05):
        super().__init__()
        self.supervised_coefficient = float(supervised_coefficient)
        self.regularization_coefficient = float(regularization_coefficient)
        self.label_smoothing = float(label_smoothing)
        weight = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32)
        self.register_buffer("class_weight", weight)

    def forward(self, output: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        supervised = F.cross_entropy(
            output["logits"], target, weight=self.class_weight,
            label_smoothing=self.label_smoothing,
        )
        return output["dmgi_loss"] + self.regularization_coefficient*output["consensus_loss"] + self.supervised_coefficient*supervised


class ImageNetResNet18Encoder(nn.Module):
    """Image encoder shared by source-informed image-tabular baselines.

    ``pretrained`` is explicit so unit tests and CPU synthetic tests never trigger
    a network download.  Formal runs use the fixed ImageNet weight enum and
    record that decision in their run configuration.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        self.output_dim = int(network.fc.in_features)
        network.fc = nn.Identity()
        self.network = network
        self.register_buffer("input_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network((image - self.input_mean) / self.input_std)


class _ImageTabularClassifier(nn.Module):
    """Common lightweight ResNet-18 and clinical MLP contract."""
    def __init__(self, clinical_dim: int, num_classes: int = 3, pretrained: bool = False, hidden_dim: int = 256, **_: object):
        super().__init__()
        if clinical_dim < 1:
            raise ValueError("image-tabular baselines require at least one clinical feature")
        self.image_encoder = ImageNetResNet18Encoder(pretrained=pretrained)
        self.image_projection = nn.Sequential(nn.Linear(self.image_encoder.output_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(True))
        self.clinical_projection = nn.Sequential(nn.Linear(clinical_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(True))
        self.num_classes = int(num_classes)


class HyperFusionClassifier(_ImageTabularClassifier):
    """Independent port of HyperFusion's tabular-conditioned hypernetwork.

    The tabular branch predicts feature-wise affine parameters for the image
    representation; no author source is copied because the upstream repository
    has no license.
    """
    def __init__(self, clinical_dim: int, num_classes: int = 3, **kwargs: object):
        super().__init__(clinical_dim, num_classes, **kwargs)
        hidden = self.image_projection[0].out_features
        self.hypernetwork = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(True), nn.Linear(hidden, 2 * hidden))
        self.classifier = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(True), nn.Dropout(.3), nn.Linear(hidden, num_classes))

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        image_feature, clinical_feature = self.image_projection(self.image_encoder(image)), self.clinical_projection(clinical)
        scale, bias = self.hypernetwork(clinical_feature).chunk(2, dim=1)
        conditioned = image_feature * (1.0 + torch.tanh(scale)) + bias
        return self.classifier(torch.cat([conditioned, clinical_feature], 1))


class AMFMedITClassifier(_ImageTabularClassifier):
    """Independent, MIT-source-informed AMF-MedIT image-tabular classifier."""
    def __init__(self, clinical_dim: int, num_classes: int = 3, alignment_weight: float = .05, **kwargs: object):
        super().__init__(clinical_dim, num_classes, **kwargs)
        hidden = self.image_projection[0].out_features
        self.image_modulation = nn.Linear(hidden, hidden)
        self.clinical_modulation = nn.Linear(hidden, hidden)
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(True), nn.Dropout(.3), nn.Linear(hidden, num_classes))
        self.alignment_weight = float(alignment_weight)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_feature, clinical_feature = self.image_projection(self.image_encoder(image)), self.clinical_projection(clinical)
        image_modulated = self.image_modulation(image_feature)
        clinical_modulated = self.clinical_modulation(clinical_feature)
        gate = self.gate(torch.cat([image_modulated, clinical_modulated], 1))
        fused = gate * image_modulated + (1.0 - gate) * clinical_modulated
        alignment = F.mse_loss(F.normalize(image_modulated, dim=1), F.normalize(clinical_modulated, dim=1))
        magnitude = (image_modulated.norm(dim=1).mean() - clinical_modulated.norm(dim=1).mean()).abs()
        return {"logits": self.classifier(fused), "aux_loss": self.alignment_weight * (alignment + .1 * magnitude)}


class MAPClassifier(_ImageTabularClassifier):
    """Independent MAP-style multimodal alignment and prediction baseline."""
    def __init__(self, clinical_dim: int, num_classes: int = 3, alignment_weight: float = .05, **kwargs: object):
        super().__init__(clinical_dim, num_classes, **kwargs)
        hidden = self.image_projection[0].out_features
        self.cross_gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(True), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(True), nn.Dropout(.3), nn.Linear(hidden, num_classes))
        self.alignment_weight = float(alignment_weight)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_feature, clinical_feature = self.image_projection(self.image_encoder(image)), self.clinical_projection(clinical)
        gate = self.cross_gate(torch.cat([image_feature, clinical_feature], 1))
        image_aligned = gate * image_feature + (1.0 - gate) * clinical_feature
        clinical_aligned = gate * clinical_feature + (1.0 - gate) * image_feature
        alignment = 1.0 - F.cosine_similarity(image_aligned, clinical_aligned, dim=1).mean()
        return {"logits": self.classifier(torch.cat([image_aligned, clinical_aligned], 1)), "aux_loss": self.alignment_weight * alignment}
