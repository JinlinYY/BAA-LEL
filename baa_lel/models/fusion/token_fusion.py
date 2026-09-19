"""Fair fusion replacements operating on identical morphology/clinical tokens."""
import torch
import torch.nn as nn

from baa_lel.models.fusion.morph_clinical_graph import MorphClinicalHeteroGraph


class TokenFusion(nn.Module):
    MODES = ("concat", "gate", "cross_attn", "heterog")

    def __init__(self, mode="heterog", morph_dim=256, clinical_dim=128,
                 hidden_dim=256, out_dim=256, dropout=0.3):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode}")
        self.mode = mode
        if mode == "heterog":
            self.heterog = MorphClinicalHeteroGraph(
                morph_dim=morph_dim, clinical_dim=clinical_dim, hidden_dim=hidden_dim,
                out_dim=out_dim, num_layers=2, dropout=dropout,
                use_residual_global=True, num_morph_nodes=5,
            )
        else:
            self.morph_proj = nn.Linear(morph_dim, hidden_dim)
            self.clin_proj = nn.Linear(clinical_dim, hidden_dim)
            self.global_proj = nn.Linear(morph_dim + clinical_dim, out_dim)
            if mode == "concat":
                self.concat = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
                                            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
            elif mode == "gate":
                self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
                self.gate_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))
            else:
                self.morph_from_clin = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
                self.clin_from_morph = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
                self.cross_out = nn.Sequential(nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, out_dim))

    def forward(self, morph_nodes, clinical_nodes, morph_global=None, clinical_global=None):
        if morph_nodes.ndim != 3 or morph_nodes.shape[1] != 5:
            raise ValueError("fusion comparison requires exactly five morphology tokens")
        if clinical_nodes.ndim != 3:
            raise ValueError("clinical_nodes must be [B,N,C]")
        if self.mode == "heterog":
            result = self.heterog(morph_nodes, clinical_nodes, morph_global, clinical_global)
            return {"fused": result["hetero_global"], **result}

        morph = self.morph_proj(morph_nodes)
        clinical = self.clin_proj(clinical_nodes)
        m_pool, c_pool = morph.mean(dim=1), clinical.mean(dim=1)
        if self.mode == "concat":
            fused = self.concat(torch.cat([m_pool, c_pool], dim=-1))
        elif self.mode == "gate":
            gate = torch.sigmoid(self.gate(torch.cat([m_pool, c_pool], dim=-1)))
            fused = self.gate_out(gate * m_pool + (1.0 - gate) * c_pool)
        else:
            morph_cross, m_attn = self.morph_from_clin(morph, clinical, clinical)
            clin_cross, c_attn = self.clin_from_morph(clinical, morph, morph)
            fused = self.cross_out(torch.cat([
                (morph + morph_cross).mean(dim=1),
                (clinical + clin_cross).mean(dim=1),
            ], dim=-1))
        if morph_global is not None and clinical_global is not None:
            fused = fused + self.global_proj(torch.cat([morph_global, clinical_global], dim=-1))
        result = {"fused": fused}
        if self.mode == "gate":
            result["gate"] = gate
        elif self.mode == "cross_attn":
            result.update({"morph_cross_attn": m_attn, "clinical_cross_attn": c_attn})
        return result
