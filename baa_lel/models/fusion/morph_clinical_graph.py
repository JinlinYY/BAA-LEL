import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphClinicalHeteroGraph(nn.Module):
    """
    Morphology-clinical heterogeneous graph fusion.

    Supports three zonal morphology nodes or five nodes including geometry and ambiguity evidence.

    Inputs:
        morph_nodes:
            [B, Nm, morph_dim]

            When Nm = 3:
                nodes = core, boundary appearance, peritumor

            When Nm = 5:
                nodes = core, boundary appearance, peritumor,
                        boundary geometry, boundary uncertainty evidence

        clinical_nodes:
            [B, Nc, clinical_dim]
            nodes = prior graph nodes initialized from the clinical vector

        morph_global:
            optional [B, morph_dim]
            image-level residual representation from multi-region pooling

        clinical_global:
            optional [B, clinical_dim]
            clinical graph-level residual representation

    Output:
        hetero_global:
            [B, out_dim]
    """

    def __init__(
        self,
        morph_dim: int = 256,
        clinical_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual_global: bool = True,
        num_morph_nodes: int = 5,
    ):
        super().__init__()

        self.morph_dim = int(morph_dim)
        self.clinical_dim = int(clinical_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.use_residual_global = bool(use_residual_global)
        self.num_morph_nodes = int(num_morph_nodes)

        if self.num_morph_nodes < 1:
            raise ValueError(f"num_morph_nodes should be positive, got {self.num_morph_nodes}")


        self.morph_proj = nn.Sequential(
            nn.LayerNorm(self.morph_dim),
            nn.Linear(self.morph_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.clin_proj = nn.Sequential(
            nn.LayerNorm(self.clinical_dim),
            nn.Linear(self.clinical_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


        self.type_embed = nn.Embedding(2, self.hidden_dim)


        morph_adj = self._build_morph_prior_adj(self.num_morph_nodes)
        self.register_buffer("morph_adj", morph_adj, persistent=False)


        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])

        self.cross_q = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_layers)
        ])
        self.cross_k = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_layers)
        ])
        self.cross_v = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_layers)
        ])


        self.readout_score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )


        if self.use_residual_global:
            self.global_proj = nn.Sequential(
                nn.LayerNorm(self.morph_dim + self.clinical_dim),
                nn.Linear(self.morph_dim + self.clinical_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.global_proj = None

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_morph_prior_adj(num_morph_nodes: int) -> torch.Tensor:
        """
        Build the fixed prior adjacency among morphology nodes.

        For the five-node boundary-evidence setting:
            0: lesion core
            1: boundary appearance
            2: peritumoral context
            3: boundary geometry token
            4: boundary uncertainty token

        The prior graph preserves anatomical continuity:
            core <-> boundary appearance <-> peritumor

        It also links boundary appearance with two boundary evidence tokens:
            boundary appearance <-> boundary geometry
            boundary appearance <-> boundary uncertainty
            boundary geometry <-> boundary uncertainty
        """
        if num_morph_nodes == 3:

            morph_adj = torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1],
                ],
                dtype=torch.float32,
            )

        elif num_morph_nodes == 5:


            morph_adj = torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1],
                    [0, 1, 0, 1, 1],
                ],
                dtype=torch.float32,
            )

        else:


            morph_adj = torch.eye(num_morph_nodes, dtype=torch.float32)
            for i in range(num_morph_nodes - 1):
                morph_adj[i, i + 1] = 1.0
                morph_adj[i + 1, i] = 1.0

        return morph_adj

    def _morph_message_passing(self, morph_h: torch.Tensor) -> torch.Tensor:
        """
        Fixed morphology-prior message passing.

        Args:
            morph_h: [B, Nm, D]

        Returns:
            [B, Nm, D]
        """
        if morph_h.dim() != 3:
            raise ValueError(f"morph_h must be [B,Nm,D], got {tuple(morph_h.shape)}")

        Nm = morph_h.size(1)
        if Nm != self.morph_adj.size(0):
            raise ValueError(
                f"morph_h node number {Nm} does not match morph_adj "
                f"size {self.morph_adj.size(0)}. Check num_morph_nodes."
            )

        adj = self.morph_adj.to(device=morph_h.device, dtype=morph_h.dtype)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj_norm = adj / deg
        return torch.einsum("ij,bjd->bid", adj_norm, morph_h)

    def _cross_type_message_passing(self, h: torch.Tensor, layer_idx: int):
        """
        Dense learnable heterogeneous message passing between all nodes.

        Args:
            h: [B, N, D], where N = Nm + Nc
        """
        q = self.cross_q[layer_idx](h)
        k = self.cross_k[layer_idx](h)
        v = self.cross_v[layer_idx](h)

        scale = h.size(-1) ** 0.5
        attn = torch.matmul(q, k.transpose(1, 2)) / scale
        attn = torch.softmax(attn, dim=-1)

        msg = torch.matmul(attn, v)
        return msg, attn

    def forward(
        self,
        morph_nodes: torch.Tensor,
        clinical_nodes: torch.Tensor,
        morph_global: torch.Tensor = None,
        clinical_global: torch.Tensor = None,
    ):
        """
        Args:
            morph_nodes:
                [B, Nm, morph_dim]
            clinical_nodes:
                [B, Nc, clinical_dim]
            morph_global:
                optional [B, morph_dim]
            clinical_global:
                optional [B, clinical_dim]
        """
        if morph_nodes.dim() != 3:
            raise ValueError(
                f"morph_nodes must be [B,Nm,C], got {tuple(morph_nodes.shape)}"
            )

        if clinical_nodes.dim() != 3:
            raise ValueError(
                f"clinical_nodes must be [B,Nc,C], got {tuple(clinical_nodes.shape)}"
            )

        B = morph_nodes.size(0)
        Nm = morph_nodes.size(1)
        Nc = clinical_nodes.size(1)

        if Nm != self.num_morph_nodes:
            raise ValueError(
                f"Expected {self.num_morph_nodes} morphology nodes, got {Nm}. "
                f"If you are using boundary geometry/uncertainty tokens, "
                f"initialize MorphClinicalHeteroGraph with num_morph_nodes=5."
            )

        if morph_nodes.size(-1) != self.morph_dim:
            raise ValueError(
                f"morph_nodes feature dim mismatch: expected {self.morph_dim}, "
                f"got {morph_nodes.size(-1)}"
            )

        if clinical_nodes.size(-1) != self.clinical_dim:
            raise ValueError(
                f"clinical_nodes feature dim mismatch: expected {self.clinical_dim}, "
                f"got {clinical_nodes.size(-1)}"
            )


        morph_h = self.morph_proj(morph_nodes)
        clin_h = self.clin_proj(clinical_nodes)

        morph_type = torch.zeros(B, Nm, dtype=torch.long, device=morph_nodes.device)
        clin_type = torch.ones(B, Nc, dtype=torch.long, device=morph_nodes.device)

        morph_h = morph_h + self.type_embed(morph_type)
        clin_h = clin_h + self.type_embed(clin_type)

        h = torch.cat([morph_h, clin_h], dim=1)

        last_attn = None


        for i in range(self.num_layers):

            morph_part = h[:, :Nm, :]
            morph_msg = self._morph_message_passing(morph_part)


            cross_msg, attn = self._cross_type_message_passing(h, i)
            last_attn = attn


            cross_msg = cross_msg.clone()
            cross_msg[:, :Nm, :] = cross_msg[:, :Nm, :] + morph_msg


            h = h + self.node_update[i](cross_msg)


        readout_logits = self.readout_score(h).squeeze(-1)
        node_attn = torch.softmax(readout_logits, dim=1)
        hetero_global = torch.sum(h * node_attn.unsqueeze(-1), dim=1)


        if self.use_residual_global and morph_global is not None and clinical_global is not None:
            if morph_global.size(-1) != self.morph_dim:
                raise ValueError(
                    f"morph_global feature dim mismatch: expected {self.morph_dim}, "
                    f"got {morph_global.size(-1)}"
                )
            if clinical_global.size(-1) != self.clinical_dim:
                raise ValueError(
                    f"clinical_global feature dim mismatch: expected {self.clinical_dim}, "
                    f"got {clinical_global.size(-1)}"
                )

            global_res = self.global_proj(torch.cat([morph_global, clinical_global], dim=1))
            hetero_global = hetero_global + global_res

        hetero_global = self.out_proj(hetero_global)


        morph_node_attn = node_attn[:, :Nm]
        clinical_node_attn = node_attn[:, Nm:]

        return {
            "hetero_global": hetero_global,
            "hetero_nodes": h,
            "hetero_node_attn": node_attn,
            "hetero_morph_node_attn": morph_node_attn,
            "hetero_clinical_node_attn": clinical_node_attn,
            "hetero_cross_attn": last_attn,
            "morph_adj": self.morph_adj,
        }
