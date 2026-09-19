import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return g
    return 1


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    return mask.clamp(0.0, 1.0)


def soft_dilate(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)


def soft_erode(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)


class RingGraphBlock(nn.Module):
    """
    Lightweight ring graph message passing:
        h_i <- MLP([h_{i-1}, h_i, h_{i+1}]) + h_i
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_left = torch.roll(h, shifts=1, dims=1)
        h_right = torch.roll(h, shifts=-1, dims=1)
        msg = torch.cat([h_left, h, h_right], dim=-1)
        return h + self.update(msg)


class AnchorConstrainedBoundaryGraph(nn.Module):
    """
    Diagnosis-Oriented Reliability-Anchored Boundary Graph Refinement.

    Inputs:
        f_seg:
            [B, 120, Hs, Ws]
            decoder-side segmentation feature

        seg_logits:
            [B, 1, Hs, Ws]
            coarse low-resolution segmentation logits

        f_cls:
            [B, 256, Hc, Wc]
            classification-oriented MedSAM semantic feature

    Outputs:
        refined_logits:
            [B, 1, Hs, Ws]

        refined_prob:
            [B, 1, Hs, Ws]

        z_geo:
            [B, token_dim]
            boundary geometry token

        z_unc:
            [B, token_dim]
            boundary uncertainty evidence token

        anchor_loss:
            scalar tensor
    """

    def __init__(
        self,
        seg_channels: int = 120,
        cls_channels: int = 256,
        token_dim: int = 256,
        num_nodes: int = 64,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        max_offset: float = 0.08,
        dropout: float = 0.1,
        eps: float = 1e-6,
        use_boundary_ambiguity: bool = True,
        use_anchor_graph: bool = True,
    ):
        super().__init__()

        self.seg_channels = int(seg_channels)
        self.cls_channels = int(cls_channels)
        self.token_dim = int(token_dim)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.max_offset = float(max_offset)
        self.eps = float(eps)
        self.use_boundary_ambiguity = bool(use_boundary_ambiguity)
        self.use_anchor_graph = bool(use_anchor_graph)


        node_in_dim = seg_channels + 1 + 1 + 2

        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_in_dim),
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gnn = nn.ModuleList([
            RingGraphBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        self.reliability_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )


        self.graph_context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, seg_channels),
            nn.GELU(),
        )

        refine_in_ch = seg_channels + 3
        self.refine_head = nn.Sequential(
            nn.Conv2d(refine_in_ch, seg_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_valid_gn_groups(seg_channels), seg_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(seg_channels, 1, kernel_size=1, bias=True),
        )


        self.geo_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 2),
            nn.Linear(hidden_dim * 2 + 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


        self.unc_readout = nn.Sequential(
            nn.LayerNorm(cls_channels * 3),
            nn.Linear(cls_channels * 3, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _make_normalized_grid(
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H * 2.0 - 1.0
        x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W * 2.0 - 1.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx.view(1, 1, H, W).expand(B, 1, H, W)
        yy = yy.view(1, 1, H, W).expand(B, 1, H, W)
        return xx, yy

    def _sample_features(self, feat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        feat: [B,C,H,W]
        q:    [B,K,2], normalized xy in [-1,1]

        return:
            [B,K,C]
        """
        grid = q.view(q.size(0), q.size(1), 1, 2)
        sampled = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def _sample_boundary_nodes(
        self,
        prob: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary_score: torch.Tensor,
    ):
        """
        Select K boundary graph nodes by uncertainty-aware boundary score,
        then sort them according to polar angle around soft lesion centroid.
        """
        B, _, H, W = prob.shape
        K = min(self.num_nodes, H * W)

        node_score = boundary_score * (0.5 + uncertainty)
        node_score = node_score.flatten(1)

        _, topk_idx = torch.topk(node_score, k=K, dim=1, largest=True, sorted=False)

        y_idx = torch.div(topk_idx, W, rounding_mode="floor")
        x_idx = topk_idx % W

        x_norm = (x_idx.to(prob.dtype) + 0.5) / W * 2.0 - 1.0
        y_norm = (y_idx.to(prob.dtype) + 0.5) / H * 2.0 - 1.0

        q = torch.stack([x_norm, y_norm], dim=-1)


        xx, yy = self._make_normalized_grid(B, H, W, prob.device, prob.dtype)
        denom = prob.sum(dim=(2, 3)).clamp_min(self.eps)
        cx = (prob * xx).sum(dim=(2, 3)) / denom
        cy = (prob * yy).sum(dim=(2, 3)) / denom

        angle = torch.atan2(q[..., 1] - cy, q[..., 0] - cx)
        order = torch.argsort(angle, dim=1)

        q = torch.gather(q, dim=1, index=order.unsqueeze(-1).expand(-1, -1, 2))
        topk_idx = torch.gather(topk_idx, dim=1, index=order)

        return q, topk_idx

    def _masked_stats_pool(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feat: [B,C,H,W]
        mask: [B,1,H,W]

        output: [B,3C] = mean + max + std
        """
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(
                mask,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        mask = _safe_mask(mask)
        B, C, H, W = feat.shape

        denom = mask.sum(dim=(2, 3)).clamp_min(self.eps)
        mean = (feat * mask).sum(dim=(2, 3)) / denom

        var = (((feat - mean.view(B, C, 1, 1)) ** 2) * mask).sum(dim=(2, 3)) / denom
        std = torch.sqrt(var.clamp_min(self.eps))

        valid = mask > self.eps
        valid_expand = valid.expand_as(feat)
        neg_inf = torch.finfo(feat.dtype).min
        masked_feat = feat.masked_fill(~valid_expand, neg_inf)
        max_val = masked_feat.flatten(2).max(dim=2).values

        has_valid = valid.flatten(2).any(dim=2)
        max_val = torch.where(has_valid.expand_as(max_val), max_val, torch.zeros_like(max_val))

        return torch.cat([mean, max_val, std], dim=1)

    def forward(
        self,
        f_seg: torch.Tensor,
        seg_logits: torch.Tensor,
        f_cls: torch.Tensor,
    ):
        if f_seg.dim() != 4:
            raise ValueError(f"f_seg must be [B,C,H,W], got {tuple(f_seg.shape)}")

        if seg_logits.dim() != 4 or seg_logits.size(1) != 1:
            raise ValueError(f"seg_logits must be [B,1,H,W], got {tuple(seg_logits.shape)}")

        if f_seg.shape[-2:] != seg_logits.shape[-2:]:
            raise ValueError(
                f"f_seg and seg_logits should have same spatial size, "
                f"got {tuple(f_seg.shape[-2:])} vs {tuple(seg_logits.shape[-2:])}"
            )

        B, _, H, W = seg_logits.shape

        prob = torch.sigmoid(seg_logits)
        uncertainty_raw = (1.0 - torch.abs(2.0 * prob - 1.0)).clamp(0.0, 1.0)
        uncertainty = uncertainty_raw if self.use_boundary_ambiguity else torch.zeros_like(uncertainty_raw)

        dil = soft_dilate(prob, kernel_size=3)
        ero = soft_erode(prob, kernel_size=3)
        boundary_score = (dil - ero).clamp(0.0, 1.0)


        if self.use_anchor_graph:
            q, node_idx = self._sample_boundary_nodes(
                prob=prob, uncertainty=uncertainty, boundary_score=boundary_score,
            )
            f_node = self._sample_features(f_seg, q)
            p_node = self._sample_features(prob, q)
            u_node = self._sample_features(uncertainty, q)
            h = self.node_proj(torch.cat([f_node, p_node, u_node, q], dim=-1))
            for block in self.gnn:
                h = block(h)
            reliability_from_prob = (1.0 - u_node).clamp(0.0, 1.0)
            reliability_learned = torch.sigmoid(self.reliability_head(h))
            reliability = (0.5 * reliability_from_prob + 0.5 * reliability_learned).clamp(0.0, 1.0)
            delta_q = torch.tanh(self.offset_head(h)) * self.max_offset
            q_ref = (q + (1.0 - reliability) * delta_q).clamp(-1.0, 1.0)
            anchor_loss = (reliability.squeeze(-1) * (q_ref - q).pow(2).sum(dim=-1)).mean()
            graph_context = self.graph_context_proj(h.mean(dim=1)).view(B, self.seg_channels, 1, 1)
            h_mean = h.mean(dim=1)
            h_std = torch.sqrt(h.var(dim=1, unbiased=False).clamp_min(self.eps))
            z_geo = self.geo_readout(torch.cat([
                h_mean, h_std, reliability.mean(dim=1), u_node.mean(dim=1)
            ], dim=-1))
            inspection = {
                "q": q.detach(), "q_ref": q_ref.detach(),
                "reliability": reliability.detach(), "node_idx": node_idx.detach(),
            }
        else:


            graph_context = torch.zeros(
                B, self.seg_channels, 1, 1, device=f_seg.device, dtype=f_seg.dtype
            )
            anchor_loss = f_seg.sum() * 0.0
            z_geo = f_cls.new_zeros((B, self.token_dim))
            inspection = {}


        refine_input = torch.cat(
            [
                f_seg + graph_context,
                prob,
                uncertainty,
                boundary_score,
            ],
            dim=1,
        )

        residual_logits = self.refine_head(refine_input)
        refined_logits = seg_logits + residual_logits
        refined_prob = torch.sigmoid(refined_logits)


        unc_boundary_mask = (boundary_score * uncertainty).clamp(0.0, 1.0)
        unc_stats = self._masked_stats_pool(f_cls, unc_boundary_mask)
        z_unc = self.unc_readout(unc_stats)
        if not self.use_boundary_ambiguity:
            z_unc = torch.zeros_like(z_unc)

        return {
            "refined_logits": refined_logits,
            "refined_prob": refined_prob,
            "z_geo": z_geo,
            "z_unc": z_unc,
            "anchor_loss": anchor_loss,
            "boundary_uncertainty": uncertainty,
            "boundary_score": boundary_score,
            "inspection": inspection,
        }
