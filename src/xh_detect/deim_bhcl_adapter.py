from __future__ import annotations

import copy
import math

import torch
from engine.core import register
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.denoising import get_contrastive_denoising_training_group
from engine.deim.dfine_decoder import (
    MLP,
    DFINETransformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from engine.deim.dfine_utils import distance2bbox, weighting_function
from engine.deim.utils import inverse_sigmoid
from torch import Tensor, nn
from torch.nn import functional as F

from .deim_bhcl import (
    HierarchicalPrototypeBank,
    hierarchical_contrastive_loss,
    xh25_hierarchy,
)


def _copy_attention_into_joint(
    source: nn.MultiheadAttention,
    destination: nn.MultiheadAttention,
) -> None:
    """Duplicate a pretrained d-dimensional attention into two 2d blocks."""

    source_dim = source.embed_dim
    joint_dim = destination.embed_dim
    if joint_dim != 2 * source_dim:
        raise ValueError("joint self-attention must have twice the source dimension")

    with torch.no_grad():
        destination.in_proj_weight.zero_()
        for projection in range(3):
            source_start = projection * source_dim
            destination_start = projection * joint_dim
            source_weight = source.in_proj_weight[source_start : source_start + source_dim]
            destination.in_proj_weight[
                destination_start : destination_start + source_dim, :source_dim
            ].copy_(source_weight)
            destination.in_proj_weight[
                destination_start + source_dim : destination_start + joint_dim,
                source_dim:,
            ].copy_(source_weight)

        if source.in_proj_bias is not None:
            assert destination.in_proj_bias is not None
            destination.in_proj_bias.zero_()
            for projection in range(3):
                source_start = projection * source_dim
                destination_start = projection * joint_dim
                source_bias = source.in_proj_bias[source_start : source_start + source_dim]
                destination.in_proj_bias[destination_start : destination_start + source_dim].copy_(
                    source_bias
                )
                destination.in_proj_bias[
                    destination_start + source_dim : destination_start + joint_dim
                ].copy_(source_bias)

        destination.out_proj.weight.zero_()
        destination.out_proj.weight[:source_dim, :source_dim].copy_(source.out_proj.weight)
        destination.out_proj.weight[source_dim:, source_dim:].copy_(source.out_proj.weight)
        if source.out_proj.bias is not None:
            assert destination.out_proj.bias is not None
            destination.out_proj.bias.copy_(source.out_proj.bias.repeat(2))


class DecoupledTransformerDecoderLayer(nn.Module):
    """Paper-faithful shared self-attention followed by task-specific streams."""

    def __init__(self, source: TransformerDecoderLayer, num_heads: int) -> None:
        super().__init__()
        hidden_dim = int(source.norm1.normalized_shape[0])
        joint_dim = 2 * hidden_dim
        device = source.self_attn.in_proj_weight.device
        dtype = source.self_attn.in_proj_weight.dtype

        # Doubling the head count preserves the pretrained per-head width.
        self.self_attn = nn.MultiheadAttention(
            joint_dim,
            2 * num_heads,
            dropout=source.self_attn.dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        _copy_attention_into_joint(source.self_attn, self.self_attn)
        self.dropout1 = copy.deepcopy(source.dropout1)
        self.norm1 = nn.LayerNorm(
            joint_dim,
            eps=source.norm1.eps,
            elementwise_affine=source.norm1.elementwise_affine,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            if source.norm1.elementwise_affine:
                self.norm1.weight.copy_(source.norm1.weight.repeat(2))
                self.norm1.bias.copy_(source.norm1.bias.repeat(2))

        self.loc_cross_attn = source.cross_attn
        self.loc_dropout2 = source.dropout2
        self.loc_gateway = source.gateway
        self.loc_linear1 = source.linear1
        self.loc_activation = source.activation
        self.loc_dropout3 = source.dropout3
        self.loc_linear2 = source.linear2
        self.loc_dropout4 = source.dropout4
        self.loc_norm3 = source.norm3

        self.cls_cross_attn = copy.deepcopy(source.cross_attn)
        self.cls_dropout2 = copy.deepcopy(source.dropout2)
        self.cls_gateway = copy.deepcopy(source.gateway)
        self.cls_linear1 = copy.deepcopy(source.linear1)
        self.cls_activation = copy.deepcopy(source.activation)
        self.cls_dropout3 = copy.deepcopy(source.dropout3)
        self.cls_linear2 = copy.deepcopy(source.linear2)
        self.cls_dropout4 = copy.deepcopy(source.dropout4)
        self.cls_norm3 = copy.deepcopy(source.norm3)

    @staticmethod
    def _with_pos_embed(tensor: Tensor, position: Tensor | None) -> Tensor:
        return tensor if position is None else tensor + position

    def _task_stream(
        self,
        target: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes,
        query_pos_embed: Tensor | None,
        *,
        prefix: str,
    ) -> Tensor:
        cross_attn = getattr(self, f"{prefix}_cross_attn")
        dropout2 = getattr(self, f"{prefix}_dropout2")
        gateway = getattr(self, f"{prefix}_gateway")
        linear1 = getattr(self, f"{prefix}_linear1")
        activation = getattr(self, f"{prefix}_activation")
        dropout3 = getattr(self, f"{prefix}_dropout3")
        linear2 = getattr(self, f"{prefix}_linear2")
        dropout4 = getattr(self, f"{prefix}_dropout4")
        norm3 = getattr(self, f"{prefix}_norm3")

        cross_output = cross_attn(
            self._with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )
        target = gateway(target, dropout2(cross_output))
        ffn_output = linear2(dropout3(activation(linear1(target))))
        return norm3((target + dropout4(ffn_output)).clamp(min=-65504, max=65504))

    def forward(
        self,
        localization_queries: Tensor,
        classification_queries: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes,
        attn_mask: Tensor | None = None,
        query_pos_embed: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        joint_queries = torch.cat([localization_queries, classification_queries], dim=-1)
        joint_position = (
            None
            if query_pos_embed is None
            else torch.cat([query_pos_embed, query_pos_embed], dim=-1)
        )
        query = key = self._with_pos_embed(joint_queries, joint_position)
        aligned, _ = self.self_attn(
            query,
            key,
            value=joint_queries,
            attn_mask=attn_mask,
        )
        joint_queries = self.norm1(joint_queries + self.dropout1(aligned))
        localization_queries, classification_queries = joint_queries.chunk(2, dim=-1)

        localization_queries = self._task_stream(
            localization_queries,
            reference_points,
            value,
            spatial_shapes,
            query_pos_embed,
            prefix="loc",
        )
        classification_queries = self._task_stream(
            classification_queries,
            reference_points,
            value,
            spatial_shapes,
            query_pos_embed,
            prefix="cls",
        )
        return localization_queries, classification_queries


class DecoupledTransformerDecoder(TransformerDecoder):
    decoupled_ready: bool

    def enable_decoupled_queries(self) -> None:
        if getattr(self, "decoupled_ready", False):
            return
        self.layers = nn.ModuleList(
            [DecoupledTransformerDecoderLayer(layer, self.num_head) for layer in self.layers]
        )
        self.decoupled_ready = True

    def forward_decoupled(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        return_classification_queries=False,
    ):
        if not getattr(self, "decoupled_ready", False):
            raise RuntimeError("decoupled decoder was not initialized before forward")

        localization_queries = target
        classification_queries = target.clone()
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_classification_queries = []
        project = (
            weighting_function(self.reg_max, up, reg_scale)
            if not hasattr(self, "project")
            else self.project
        )
        ref_points_detach = torch.sigmoid(ref_points_unact)

        for layer_index, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if layer_index >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(
                    memory,
                    None,
                    query_pos_embed.shape[-1],
                    memory_mask,
                    spatial_shapes,
                )
                localization_queries = F.interpolate(
                    localization_queries, size=query_pos_embed.shape[-1]
                )
                classification_queries = F.interpolate(
                    classification_queries, size=query_pos_embed.shape[-1]
                )
                output_detach = localization_queries.detach()

            localization_queries, classification_queries = layer(
                localization_queries,
                classification_queries,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )

            if layer_index == 0:
                pre_bboxes = torch.sigmoid(
                    pre_bbox_head(localization_queries) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](classification_queries)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = (
                bbox_head[layer_index](localization_queries + output_detach) + pred_corners_undetach
            )
            inter_ref_bbox = distance2bbox(
                ref_points_initial,
                integral(pred_corners, project),
                reg_scale,
            )

            if self.training or layer_index == self.eval_idx:
                scores = score_head[layer_index](classification_queries)
                scores = self.lqe_layers[layer_index](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if return_classification_queries:
                    dec_out_classification_queries.append(classification_queries)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = localization_queries.detach()

        query_outputs = (
            torch.stack(dec_out_classification_queries) if dec_out_classification_queries else None
        )
        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
            query_outputs,
        )


@register()
class BHCLDFINETransformer(DFINETransformer):
    """D-FINE decoder with the paper's classification/localization query split."""

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=None,
        feat_strides=None,
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        decouple_queries=True,
        bhcl_mode="none",
        bhcl_embedding_dim=128,
    ) -> None:
        if bhcl_mode not in {"none", "hcl", "bhcl"}:
            raise ValueError("bhcl_mode must be none, hcl, or bhcl")
        if bhcl_mode != "none" and not decouple_queries:
            raise ValueError("HCL/BHCL requires decoupled classification queries")
        if bhcl_embedding_dim <= 0:
            raise ValueError("bhcl_embedding_dim must be positive")

        feat_channels = [512, 1024, 2048] if feat_channels is None else list(feat_channels)
        feat_strides = [8, 16, 32] if feat_strides is None else list(feat_strides)

        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
        )
        self.decouple_queries = decouple_queries
        self.bhcl_mode = bhcl_mode
        self.bhcl_embedding_dim = bhcl_embedding_dim
        self.decoder.__class__ = DecoupledTransformerDecoder
        self.decoder.decoupled_ready = False

        if bhcl_mode != "none":
            layer_dims = [int(layer.norm1.normalized_shape[0]) for layer in self.decoder.layers]
            if len(set(layer_dims)) != 1:
                raise ValueError(
                    "BHCL requires equal decoder widths for one shared projection space"
                )
            self.bhcl_projection_head = MLP(
                layer_dims[0],
                layer_dims[0],
                bhcl_embedding_dim,
                2,
                act=mlp_act,
            )

    def initialize_after_tuning(self) -> None:
        if self.decouple_queries:
            self.decoder.enable_decoupled_queries()

    def convert_to_deploy(self) -> None:
        # FLOP profilers attach hooks before the first forward pass, so the
        # decoder must be structurally final before those hooks are registered.
        self.initialize_after_tuning()
        super().convert_to_deploy()
        if hasattr(self, "bhcl_projection_head"):
            self.bhcl_projection_head = nn.Identity()

    def _project_classification_queries(self, queries: Tensor) -> Tensor:
        return F.normalize(self.bhcl_projection_head(queries), dim=-1)

    def forward(self, feats, targets=None):
        self.initialize_after_tuning()
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits = denoising_bbox_unact = attn_mask = dn_meta = None

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )

        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            classification_queries,
        ) = self.decoder.forward_decoupled(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            return_classification_queries=self.training and self.bhcl_mode != "none",
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            if classification_queries is not None:
                _, classification_queries = torch.split(
                    classification_queries, dn_meta["dn_num_split"], dim=2
                )

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        projected_queries = None
        if classification_queries is not None:
            projected_queries = self._project_classification_queries(classification_queries)
            out["bhcl_features"] = projected_queries[-1]

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            if projected_queries is not None:
                for aux_output, features in zip(
                    out["aux_outputs"], projected_queries[:-1], strict=True
                ):
                    aux_output["bhcl_features"] = features
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_bboxes,
            }
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta

        return out


@register()
class BHCLDEIMCriterion(DEIMCriterion):
    """DEIM criterion extension for per-decoder-layer HCL and BHCL."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        bhcl_mode="bhcl",
        bhcl_embedding_dim=128,
        bhcl_temperature=0.1,
        bhcl_epsilon=0.1,
    ) -> None:
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set,
        )
        if bhcl_mode not in {"hcl", "bhcl"}:
            raise ValueError("bhcl_mode must be hcl or bhcl")
        if "bhcl" not in losses or "loss_bhcl" not in weight_dict:
            raise ValueError("BHCL criterion requires bhcl loss and loss_bhcl weight")
        if bhcl_temperature <= 0 or not math.isfinite(bhcl_temperature):
            raise ValueError("bhcl_temperature must be finite and positive")

        self.bhcl_mode = bhcl_mode
        self.bhcl_temperature = bhcl_temperature
        self.hierarchy = xh25_hierarchy()
        if self.hierarchy.num_leaf_classes != num_classes:
            raise ValueError("XH25 BHCL hierarchy requires exactly 25 classes")
        self.prototype_bank = (
            HierarchicalPrototypeBank(
                self.hierarchy,
                bhcl_embedding_dim,
                epsilon=bhcl_epsilon,
            )
            if bhcl_mode == "bhcl"
            else None
        )
        self._prototype_updates: list[tuple[Tensor, Tensor]] = []

    def loss_bhcl(self, outputs, targets, indices, num_boxes):
        del num_boxes
        if "bhcl_features" not in outputs:
            return {}
        source_index = self._get_src_permutation_idx(indices)
        features = outputs["bhcl_features"][source_index]
        labels = torch.cat(
            [
                target["labels"][target_index]
                for target, (_, target_index) in zip(targets, indices, strict=True)
            ]
        )
        prototypes = (
            self.prototype_bank.by_level(dtype=features.dtype)
            if self.prototype_bank is not None
            else None
        )
        if self.bhcl_mode == "hcl":
            loss = hierarchical_contrastive_loss(
                features,
                labels,
                self.hierarchy,
                mode="hcl",
                temperature=self.bhcl_temperature,
            )
        else:
            loss = hierarchical_contrastive_loss(
                features,
                labels,
                self.hierarchy,
                mode="bhcl",
                temperature=self.bhcl_temperature,
                prototypes=prototypes,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("BHCL became non-finite before DEIM loss reduction")
        if self.training and self.prototype_bank is not None:
            self._prototype_updates.append((features.detach(), labels.detach()))
        return {"loss_bhcl": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "bhcl":
            return self.loss_bhcl(outputs, targets, indices, num_boxes)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        self._prototype_updates = []
        losses = super().forward(outputs, targets, **kwargs)
        if self.training and self.prototype_bank is not None and self._prototype_updates:
            features = torch.cat([item[0] for item in self._prototype_updates])
            labels = torch.cat([item[1] for item in self._prototype_updates])
            self.prototype_bank.update(features, labels)
        self._prototype_updates = []
        return losses
