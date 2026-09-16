import torch
import torch.nn as nn
import torch.nn.functional as F

from bua_lel.models.heads import LesionSegmentationDecoder, EnhancedClassifier
from bua_lel.models.morphology.zonal_morphology import ZonalMorphologyEncoder
from bua_lel.models.backbones.medsam import MedSAMMultiScaleEncoder
from bua_lel.models.backbones.resnet50_encoder import ResNet50MultiScaleEncoder
from bua_lel.models.clinical.clinical_graph import ClinicalVariableGraphEncoder
from bua_lel.models.fusion.morph_clinical_graph import MorphClinicalHeteroGraph
from bua_lel.models.boundary import AnchorConstrainedBoundaryGraph
from bua_lel.models.fusion.token_fusion import TokenFusion


class BUALEL(nn.Module):
    """Joint lesion segmentation and morphology-clinical evidence learning.

    Returns segmentation logits, class logits, and an evidence dictionary.
    The image encoder, boundary graph, zonal morphology encoder, and
    heterogeneous graph are jointly optimized using the configured objective.
    """

    def __init__(
        self,
        clinical_dim: int,
        numeric_slice=None,
        onehot_slices_dict=None,
        num_classes: int = 3,
        use_pca: bool = False,
        pca_dim: int = 100,
        clin_embed_dim: int = 128,
        cls_hidden_dims=(512, 256),
        cls_dropout: float = 0.3,
        detach_roi_in_cls: bool = True,
        detach_segfeat_in_cls: bool = False,
        lambda_cons: float = 0.0,
        drop_path_rate: float = 0.1,
        se_ratio: float = 0.25,
        gn_groups: int = 32,


        medsam_checkpoint_path=None,
        medsam_model_type: str = "vit_b",
        freeze_medsam: bool = True,
        unfreeze_medsam_last_n: int = 0,
        medsam_activation_checkpointing: bool = False,
        image_size: int = 1024,


        clinical_graph_dim: int = 128,
        clinical_graph_hidden_dim: int = 128,
        clinical_graph_dropout: float = 0.1,
        clinical_graph_layers: int = 3,
        clinical_graph_pooling: str = "mean",
        use_causal_graph: bool = True,
        add_reverse_edges: bool = True,
        add_self_loops: bool = True,


        hetero_hidden_dim: int = 256,
        hetero_out_dim: int = 256,
        hetero_layers: int = 2,


        use_boundary_refiner: bool = True,
        boundary_num_nodes: int = 64,
        boundary_hidden_dim: int = 128,
        boundary_gnn_layers: int = 3,
        boundary_max_offset: float = 0.08,
        lambda_anchor: float = 0.01,
        boundary_token_gate_init: float = -2.0,


        boundary_radius_ratio: float = 0.15,
        peritumor_radius_ratio: float = 0.40,
        min_boundary_radius: int = 2,
        max_boundary_radius: int = 4,
        min_peritumor_radius: int = 4,
        max_peritumor_radius: int = 8,
        modality: str = "multimodal",
        enabled_regions=("core", "boundary", "peritumoral"),
        use_z_geo: bool = True,
        use_z_unc: bool = True,
        fusion_mode: str = "heterog",
        late_raw_clinical_fusion: bool = False,
        classifier_fused_dim: int = 256,
        use_boundary_ambiguity: bool = True,
        use_anchor_graph: bool = True,
        use_zonal_evidence: bool = True,
        architecture_profile: str = "extended",
        visual_backbone: str = "medsam_vit_b",
        visual_checkpoint_path=None,
        backbone_unfreeze_policy: str = "last_transformer_block",
    ):
        super().__init__()

        self.clinical_dim = int(clinical_dim)
        self.num_classes = int(num_classes)
        self.detach_roi_in_cls = bool(detach_roi_in_cls)
        self.detach_segfeat_in_cls = bool(detach_segfeat_in_cls)


        self.lambda_cons = 0.0
        self.use_boundary_refiner = bool(use_boundary_refiner)
        self.lambda_anchor = float(lambda_anchor)
        if modality not in ("ultrasound_only", "clinical_only", "multimodal"):
            raise ValueError(f"Unknown modality: {modality}")
        if late_raw_clinical_fusion and modality == "ultrasound_only":
            raise ValueError(
                "late_raw_clinical_fusion cannot be enabled for ultrasound_only modality"
            )
        if self.clinical_dim < 0:
            raise ValueError("clinical_dim must be non-negative")
        if self.clinical_dim == 0 and modality != "ultrasound_only":
            raise ValueError(
                "clinical_dim=0 is supported only for ultrasound_only modality"
            )
        self.modality = modality
        self.enabled_regions = tuple(enabled_regions)
        unknown_regions = set(self.enabled_regions) - {"core", "boundary", "peritumoral"}
        if unknown_regions or not self.enabled_regions:
            raise ValueError(f"Invalid enabled_regions: {self.enabled_regions}")
        self.use_z_geo = bool(use_z_geo)
        self.use_z_unc = bool(use_z_unc)
        self.fusion_mode = fusion_mode
        self.late_raw_clinical_fusion = bool(late_raw_clinical_fusion)
        self.classifier_fused_dim = int(classifier_fused_dim)
        if self.classifier_fused_dim <= 0:
            raise ValueError("classifier_fused_dim must be positive")
        self.use_zonal_evidence = bool(use_zonal_evidence)
        self.architecture_profile = str(architecture_profile)
        if self.architecture_profile not in {"extended", "paper"}:
            raise ValueError(
                "architecture_profile must be 'extended' or 'paper', "
                f"got {architecture_profile!r}"
            )
        self.visual_backbone = str(visual_backbone)
        self.backbone_unfreeze_policy = str(backbone_unfreeze_policy)
        if self.visual_backbone not in {
            "medsam_vit_b", "sam_vit_b", "imagenet_resnet50",
        }:
            raise ValueError(f"Unknown visual_backbone: {self.visual_backbone!r}")


        self.boundary_token_gate = nn.Parameter(
            torch.tensor(float(boundary_token_gate_init), dtype=torch.float32)
        )


        self.numeric_slice = numeric_slice
        self.onehot_slices_dict = onehot_slices_dict
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.clin_embed_dim = clin_embed_dim
        self.drop_path_rate = drop_path_rate
        self.se_ratio = se_ratio


        if self.visual_backbone in {"medsam_vit_b", "sam_vit_b"}:
            if self.backbone_unfreeze_policy != "last_transformer_block":
                raise ValueError(
                    f"{self.visual_backbone} requires last_transformer_block"
                )
            checkpoint_path = visual_checkpoint_path or medsam_checkpoint_path
            self.encoder = MedSAMMultiScaleEncoder(
                checkpoint_path=checkpoint_path,
                model_type=medsam_model_type,
                freeze_medsam=freeze_medsam,
                unfreeze_last_n=unfreeze_medsam_last_n,
                activation_checkpointing=medsam_activation_checkpointing,
                out_channels=(120, 240, 480, 960),
                gn_groups=gn_groups,
                force_img_size=image_size,
                out_indices=(4, 7, 10),
                return_cls_feat=True,
                cls_out_dim=256,
            )
        else:
            if self.backbone_unfreeze_policy != "last_bottleneck":
                raise ValueError("imagenet_resnet50 requires last_bottleneck")
            self.encoder = ResNet50MultiScaleEncoder(
                checkpoint_path=visual_checkpoint_path,
                pretrained=visual_checkpoint_path is None,
                unfreeze_last_bottleneck=True,
                out_channels=(120, 240, 480, 960),
                cls_out_dim=256,
                gn_groups=gn_groups,
            )


        self.decoder = LesionSegmentationDecoder(gn_groups=gn_groups)
        self.seg_head = nn.Conv2d(120, 1, kernel_size=1, bias=True)


        if self.use_boundary_refiner:
            self.boundary_refiner = AnchorConstrainedBoundaryGraph(
                seg_channels=120,
                cls_channels=256,
                token_dim=256,
                num_nodes=boundary_num_nodes,
                hidden_dim=boundary_hidden_dim,
                num_gnn_layers=boundary_gnn_layers,
                max_offset=boundary_max_offset,
                dropout=cls_dropout,
                use_boundary_ambiguity=use_boundary_ambiguity,
                use_anchor_graph=use_anchor_graph,
            )
        else:
            self.boundary_refiner = None


        self.roi_pool = ZonalMorphologyEncoder(
            in_channels=256,
            boundary_kernel=7,
            peritumor_kernel=17,
            use_region_attention=True,
            dropout=cls_dropout,
            binarize_roi=False,
            detach_roi=detach_roi_in_cls,
            less_overlap_peritumor=True,
            use_adaptive_kernel=True,
            boundary_radius_ratio=boundary_radius_ratio,
            peritumor_radius_ratio=peritumor_radius_ratio,
            min_boundary_radius=min_boundary_radius,
            max_boundary_radius=max_boundary_radius,
            min_peritumor_radius=min_peritumor_radius,
            max_peritumor_radius=max_peritumor_radius,
            use_small_lesion_gate=True,
            small_lesion_area_threshold=500.0,
            small_lesion_gate_temperature=150.0,
            hard_disable_peritumor_for_small=False,
            suppress_peritumor_attention=True,
        )


        self.clin_graph = (
            ClinicalVariableGraphEncoder(
                clinical_dim=self.clinical_dim,
                numeric_slice=numeric_slice,
                onehot_slices_dict=onehot_slices_dict,
                graph_dim=clinical_graph_dim,
                hidden_dim=clinical_graph_hidden_dim,
                dropout=clinical_graph_dropout,
                use_causal_graph=use_causal_graph,
                add_reverse_edges=add_reverse_edges,
                add_self_loops=add_self_loops,
                causal_graph_layers=clinical_graph_layers,
                pooling=clinical_graph_pooling,
            )
            if self.modality != "ultrasound_only"
            else None
        )


        self.num_morph_nodes = 5 if self.use_boundary_refiner else 3

        self.hetero_graph = (
            MorphClinicalHeteroGraph(
                morph_dim=256,
                clinical_dim=clinical_graph_dim,
                hidden_dim=hetero_hidden_dim,
                out_dim=hetero_out_dim,
                num_layers=hetero_layers,
                dropout=cls_dropout,
                use_residual_global=True,
                num_morph_nodes=self.num_morph_nodes,
            ) if fusion_mode == "heterog" else None
        )
        if (
            self.architecture_profile == "paper"
            and self.modality == "multimodal"
            and self.fusion_mode == "heterog"
        ):


            self.alternative_fusion = None
            self.ultrasound_only_fusion = None
            self.clinical_only_fusion = None
        else:
            self.alternative_fusion = (
                TokenFusion(
                    mode=fusion_mode, morph_dim=256, clinical_dim=clinical_graph_dim,
                    hidden_dim=hetero_hidden_dim, out_dim=hetero_out_dim, dropout=cls_dropout,
                ) if fusion_mode != "heterog" else None
            )
            self.ultrasound_only_fusion = nn.Sequential(
                nn.Linear(256, hetero_hidden_dim), nn.GELU(), nn.Dropout(cls_dropout),
                nn.Linear(hetero_hidden_dim, hetero_out_dim),
            )
            self.clinical_only_fusion = nn.Sequential(
                nn.Linear(clinical_graph_dim, hetero_hidden_dim), nn.GELU(), nn.Dropout(cls_dropout),
                nn.Linear(hetero_hidden_dim, hetero_out_dim),
            )


        self.classifier_fused_projection = (
            nn.Identity()
            if self.classifier_fused_dim == hetero_out_dim
            else nn.Sequential(
                nn.Linear(hetero_out_dim, self.classifier_fused_dim),
                nn.LayerNorm(self.classifier_fused_dim),
                nn.GELU(),
            )
        )
        classifier_feature_dim = self.classifier_fused_dim + (
            self.clinical_dim if self.late_raw_clinical_fusion else 0
        )
        self.cls_head = EnhancedClassifier(
            feature_dim=classifier_feature_dim,
            output_size=self.num_classes,
            hidden_dims=cls_hidden_dims,
            dropout_rate=cls_dropout,
        )


        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_imp = nn.Parameter(torch.tensor(0.0))


        self._init_task_weights()

    @staticmethod
    def _apply_analysis_controls(morph_nodes, analysis_controls=None):
        """Apply inference-only morphology-node masking without mutating model state.

        ``analysis_controls`` is deliberately an opt-in analysis interface.  The
        normal training/inference path passes ``None`` and returns the original
        tensor unchanged.  Currently the only supported control is
        ``{"zero_morph_nodes": [node indices]}``; the caller owns the semantic
        names of those indices so ablation labels cannot silently alter model
        architecture.  The operation creates a local multiplicative mask and
        never writes to a tensor stored in the model or to a parameter.
        """
        if analysis_controls is None:
            return morph_nodes, ()
        if not isinstance(analysis_controls, dict):
            raise TypeError("analysis_controls must be None or a dict")
        unknown = set(analysis_controls) - {"zero_morph_nodes"}
        if unknown:
            raise ValueError(f"Unsupported analysis_controls keys: {sorted(unknown)}")
        raw_nodes = analysis_controls.get("zero_morph_nodes", ())
        if raw_nodes is None:
            raw_nodes = ()
        try:
            node_indices = tuple(sorted({int(index) for index in raw_nodes}))
        except (TypeError, ValueError) as error:
            raise ValueError("zero_morph_nodes must be an iterable of integer indices") from error
        if any(index < 0 or index >= morph_nodes.size(1) for index in node_indices):
            raise ValueError(
                f"zero_morph_nodes must be within [0, {morph_nodes.size(1) - 1}], "
                f"got {node_indices}"
            )
        if not node_indices:
            return morph_nodes, node_indices
        local_mask = morph_nodes.new_ones((1, morph_nodes.size(1), 1))
        local_mask[:, list(node_indices), :] = 0.0
        return morph_nodes * local_mask, node_indices

    def forward(self, x_img, c_obs=None, m=None, task="both", analysis_controls=None):
        """Forward."""
        if x_img.dim() != 4:
            raise ValueError(f"x_img must be [B,C,H,W], got {tuple(x_img.shape)}")

        B, _, H, W = x_img.shape

        need_seg = task in ["seg", "both"]
        need_cls = task in ["cls", "both"]


        enc_out = self.encoder(x_img, return_dict=True)

        f1, f2, f3, f4 = enc_out["seg_feats"]
        cls_feat = enc_out["cls_feat"]


        feat_seg = self.decoder(f1, f2, f3, f4)

        seg_logits_low_raw = self.seg_head(feat_seg)

        if self.use_boundary_refiner:
            boundary_out = self.boundary_refiner(
                f_seg=feat_seg,
                seg_logits=seg_logits_low_raw,
                f_cls=cls_feat,
            )
            seg_logits_low = boundary_out["refined_logits"]
            seg_prob_low = boundary_out["refined_prob"]
        else:
            boundary_out = None
            seg_logits_low = seg_logits_low_raw
            seg_prob_low = torch.sigmoid(seg_logits_low)

        roi_prob = seg_prob_low

        seg_logits = None
        if need_seg:
            seg_logits = F.interpolate(
                seg_logits_low,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        cls_logits = None

        aux = {
            "roi_prob": roi_prob,
            "seg_logits_low": seg_logits_low,
            "seg_logits_low_raw": seg_logits_low_raw,
            "seg_prob_low": seg_prob_low,
            "cls_feat": cls_feat,
            "feat_seg": feat_seg,
        }

        if boundary_out is not None:
            aux.update({
                "boundary_out": boundary_out,
                "boundary_anchor_loss": boundary_out["anchor_loss"],
                "z_geo": boundary_out["z_geo"],
                "z_unc": boundary_out["z_unc"],
                "boundary_uncertainty": boundary_out["boundary_uncertainty"],
                "boundary_score": boundary_out["boundary_score"],
            })
        else:
            zero = seg_logits_low.sum() * 0.0
            aux.update({
                "boundary_out": None,
                "boundary_anchor_loss": zero,
                "z_geo": None,
                "z_unc": None,
                "boundary_uncertainty": None,
                "boundary_score": None,
            })


        if need_cls:
            if c_obs is None:
                if self.modality == "ultrasound_only" and self.clinical_dim == 0:
                    c_obs = x_img.new_zeros((B, 0))
                else:
                    raise ValueError("Classification requires c_obs.")

            if c_obs.dim() != 2 or c_obs.shape[0] != B or c_obs.shape[1] != self.clinical_dim:
                raise ValueError(
                    f"c_obs must be [B,{self.clinical_dim}] and match batch size, "
                    f"but got {tuple(c_obs.shape)}."
                )


            if m is not None:
                if m.shape[0] != B or m.shape[1] != self.clinical_dim:
                    raise ValueError(
                        f"m must be [B,{self.clinical_dim}] when provided, "
                        f"but got {tuple(m.shape)}."
                    )

            roi_for_cls = roi_prob.detach() if self.detach_roi_in_cls else roi_prob
            feat_for_cls = cls_feat.detach() if self.detach_segfeat_in_cls else cls_feat


            v_img, roi_info = self.roi_pool(
                feat_for_cls,
                roi_for_cls,
            )

            morph_nodes_3_raw = roi_info["region_feats"]
            exact_reference_regions = (
                self.architecture_profile == "paper"
                and self.enabled_regions == ("core", "boundary", "peritumoral")
                and self.use_zonal_evidence
            )
            if exact_reference_regions:

                morph_nodes_3 = morph_nodes_3_raw
            else:
                region_names = ("core", "boundary", "peritumoral")
                region_mask = morph_nodes_3_raw.new_tensor([
                    float(name in self.enabled_regions and self.use_zonal_evidence)
                    for name in region_names
                ]).view(1, 3, 1)
                morph_nodes_3 = morph_nodes_3_raw * region_mask
                if self.use_zonal_evidence:
                    selected_weights = roi_info["region_weights"] * region_mask.squeeze(-1)
                    selected_weights = selected_weights / selected_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    selected_v_region = (morph_nodes_3_raw * selected_weights.unsqueeze(-1)).sum(dim=1)
                    v_img = self.roi_pool.final_proj(
                        torch.cat([selected_v_region, roi_info["v_global"]], dim=1)
                    ) + roi_info["v_global"]
                else:
                    v_img = roi_info["v_global"]

            if self.use_boundary_refiner:
                z_geo = boundary_out["z_geo"]
                z_unc = boundary_out["z_unc"]


                if self.detach_segfeat_in_cls:
                    z_geo = z_geo.detach()
                    z_unc = z_unc.detach()


                boundary_gate = torch.sigmoid(self.boundary_token_gate)
                z_geo_gated = boundary_gate * z_geo
                z_unc_gated = boundary_gate * z_unc
                if not self.use_z_geo:
                    z_geo_gated = torch.zeros_like(z_geo_gated)
                if not self.use_z_unc:
                    z_unc_gated = torch.zeros_like(z_unc_gated)

                boundary_evidence_nodes = torch.stack(
                    [z_geo_gated, z_unc_gated],
                    dim=1,
                )

                morph_nodes = torch.cat(
                    [morph_nodes_3, boundary_evidence_nodes],
                    dim=1,
                )
            else:
                z_geo = None
                z_unc = None
                z_geo_gated = None
                z_unc_gated = None
                boundary_gate = None
                boundary_evidence_nodes = None
                morph_nodes = morph_nodes_3


            morph_nodes, zeroed_morph_nodes = self._apply_analysis_controls(
                morph_nodes, analysis_controls
            )


            if self.modality == "ultrasound_only":
                graph_out = {}
                clinical_nodes = None
                v_clin = None
            else:
                graph_out = self.clin_graph(c_obs.float(), m)
                clinical_nodes = graph_out["clinical_nodes"]
                v_clin = graph_out["clinical_global"]


            if self.modality == "ultrasound_only":
                vfused = self.ultrasound_only_fusion(morph_nodes.mean(dim=1))
                hetero_out = {}
            elif self.modality == "clinical_only":
                vfused = self.clinical_only_fusion(clinical_nodes.mean(dim=1))
                hetero_out = {}
            elif self.fusion_mode == "heterog":
                hetero_out = self.hetero_graph(
                    morph_nodes=morph_nodes, clinical_nodes=clinical_nodes,
                    morph_global=v_img, clinical_global=v_clin,
                )
                vfused = hetero_out["hetero_global"]
            else:
                hetero_out = self.alternative_fusion(
                    morph_nodes, clinical_nodes, v_img, v_clin,
                )
                vfused = hetero_out["fused"]
            classifier_input = self.compose_classifier_input(vfused, c_obs)
            classifier_fused = classifier_input[:, :self.classifier_fused_dim]
            cls_logits = self.cls_head(classifier_input)

            aux.update({

                "v_img": v_img,
                "morph_nodes": morph_nodes,
                "analysis_zeroed_morph_nodes": zeroed_morph_nodes,
                "morph_nodes_3": morph_nodes_3,
                "boundary_evidence_nodes": boundary_evidence_nodes,
                "roi_info": roi_info,
                "region_feats": morph_nodes_3,
                "region_feats_with_boundary_evidence": morph_nodes,
                "region_weights": roi_info.get("region_weights", None),
                "z_geo_cls": z_geo,
                "z_unc_cls": z_unc,
                "z_geo_gated": z_geo_gated,
                "z_unc_gated": z_unc_gated,
                "boundary_token_gate": (
                    torch.sigmoid(self.boundary_token_gate).detach()
                    if self.use_boundary_refiner else None
                ),


                "v_clin": v_clin,
                "v_clin_star": v_clin,
                "clinical_nodes": clinical_nodes,


                "vfused": vfused,
                "vfused_raw": vfused,
                "classifier_fused": classifier_fused,
                "classifier_input": classifier_input,
                "hetero_global": vfused,
                "hetero_nodes": hetero_out.get("hetero_nodes", None),
                "hetero_node_attn": hetero_out.get("hetero_node_attn", None),
                "hetero_cross_attn": hetero_out.get("hetero_cross_attn", None),
                "hetero_morph_node_attn": hetero_out.get("hetero_morph_node_attn", None),
                "hetero_clinical_node_attn": hetero_out.get("hetero_clinical_node_attn", None),

                **graph_out,
            })

        if task == "seg":
            return seg_logits, None, aux

        if task == "cls":
            return None, cls_logits, aux

        return seg_logits, cls_logits, aux

    def forward_cls_with_region_feats(self, x_img, c_obs, m, region_feats_override):
        """
        前向分类分支，用于局部扰动分析，允许替换 region_feats
        """
        B = x_img.shape[0]
        enc_out = self.encoder(x_img, return_dict=True)
        cls_feat = enc_out["cls_feat"]

        roi_prob = torch.zeros((B, 1, cls_feat.shape[2], cls_feat.shape[3]), device=x_img.device)
        v_img, roi_info = self.roi_pool(cls_feat, roi_prob)
        morph_nodes_3 = region_feats_override

        if self.use_boundary_refiner:
            boundary_out = self.boundary_refiner(f_seg=cls_feat, seg_logits=roi_prob, f_cls=cls_feat)
            z_geo_gated = torch.sigmoid(self.boundary_token_gate) * boundary_out["z_geo"]
            z_unc_gated = torch.sigmoid(self.boundary_token_gate) * boundary_out["z_unc"]
            morph_nodes = torch.cat([morph_nodes_3, torch.stack([z_geo_gated, z_unc_gated], dim=1)], dim=1)
        else:
            morph_nodes = morph_nodes_3

        if self.modality == "ultrasound_only":
            vfused = self.ultrasound_only_fusion(morph_nodes.mean(dim=1))
        else:
            graph_out = self.clin_graph(c_obs.float(), m)
            clinical_nodes = graph_out["clinical_nodes"]
            v_clin = graph_out["clinical_global"]
            if self.modality == "clinical_only":
                vfused = self.clinical_only_fusion(clinical_nodes.mean(dim=1))
            elif self.fusion_mode == "heterog":
                hetero_out = self.hetero_graph(
                    morph_nodes=morph_nodes,
                    clinical_nodes=clinical_nodes,
                    morph_global=v_img,
                    clinical_global=v_clin,
                )
                vfused = hetero_out["hetero_global"]
            else:
                hetero_out = self.alternative_fusion(
                    morph_nodes, clinical_nodes, v_img, v_clin,
                )
                vfused = hetero_out["fused"]
        classifier_input = self.compose_classifier_input(vfused, c_obs)
        cls_logits = self.cls_head(classifier_input)
        return cls_logits

    def compose_classifier_input(self, vfused, c_obs):
        """Build the classifier input, optionally adding a raw-clinical shortcut."""
        if vfused.dim() != 2:
            raise ValueError(f"vfused must be [B,C], got {tuple(vfused.shape)}")
        vfused = self.classifier_fused_projection(vfused)
        if not self.late_raw_clinical_fusion:
            return vfused
        if c_obs is None or c_obs.dim() != 2:
            shape = None if c_obs is None else tuple(c_obs.shape)
            raise ValueError(f"late raw clinical fusion requires c_obs [B,C], got {shape}")
        if c_obs.shape != (vfused.shape[0], self.clinical_dim):
            raise ValueError(
                f"late raw clinical fusion expected c_obs [{vfused.shape[0]},{self.clinical_dim}], "
                f"got {tuple(c_obs.shape)}"
            )
        clinical = c_obs.to(device=vfused.device, dtype=vfused.dtype)
        return torch.cat([vfused, clinical], dim=1)


    def get_total_loss(
        self,
        Lseg,
        Lcls,
        Limp=0.0,
        Lcons=0.0,
        Lanchor=0.0,
        lambda_anchor=None,
        clamp=(-5.0, 5.0),
    ):
        """Get total loss."""
        lo, hi = clamp

        lv_seg = self.log_var_seg.clamp(lo, hi)
        lv_cls = self.log_var_cls.clamp(lo, hi)

        if not torch.is_tensor(Limp):
            ref = Lcls if torch.is_tensor(Lcls) else Lseg
            Limp = torch.zeros((), device=ref.device, dtype=ref.dtype)

        if not torch.is_tensor(Lanchor):
            ref = Lcls if torch.is_tensor(Lcls) else Lseg
            Lanchor = torch.zeros((), device=ref.device, dtype=ref.dtype)

        if lambda_anchor is None:
            lambda_anchor = self.lambda_anchor

        loss = (
            0.5 * torch.exp(-lv_seg) * Lseg + 0.5 * lv_seg
            + 0.5 * torch.exp(-lv_cls) * Lcls + 0.5 * lv_cls
            + float(lambda_anchor) * Lanchor
        )

        weights = {
            "w_seg": float((0.5 * torch.exp(-lv_seg)).detach().cpu()),
            "w_cls": float((0.5 * torch.exp(-lv_cls)).detach().cpu()),
            "w_imp": 0.0,
            "lambda_cons": 0.0,
            "lambda_anchor": float(lambda_anchor),
            "Lanchor": float(Lanchor.detach().cpu()),
        }

        return loss, weights

    def _init_task_weights(self):
        """
        Initialize only task-specific modules.

        Skip:
            encoder.image_encoder

        because it is the pretrained MedSAM ViT image encoder loaded from
        medsam_checkpoint_path.
        """
        skip_prefixes = (
            "encoder.image_encoder",
        )

        for name, mm in self.named_modules():
            if any(name == p or name.startswith(p + ".") for p in skip_prefixes):
                continue

            if isinstance(mm, nn.Conv2d):
                nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

            elif isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight)
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

            elif isinstance(mm, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(mm, "weight") and mm.weight is not None:
                    nn.init.constant_(mm.weight, 1.0)
                if hasattr(mm, "bias") and mm.bias is not None:
                    nn.init.constant_(mm.bias, 0.0)


    def _init_basic_weights(self):
        self._init_task_weights()
