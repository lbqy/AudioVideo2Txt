# Some codes are borrowed from fairseq so that
# we can simplify the project.

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from copy import deepcopy
from omegaconf import II
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel
from fairseq.modules import GradMultiply

from .modality_encoder import ResEncoder, AudioEncoder
from .transformer_encoder import TransformerEncoder
from utils import compute_mask_indices, get_available_activation_fns

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class EncoderBackboneConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'per-sample probability of applying modality dropout during AV training'})
    audio_dropout: float = field(default=0.5, metadata={'help': 'conditional probability of dropping audio after a sample is selected for modality dropout'})
    modality_dropout_mode: str = field(default='sample', metadata={'help': 'modality dropout mode: sample drops a full modality, span drops a time span'})
    modality_dropout_span_min: int = field(default=10, metadata={'help': 'minimum span length in frames for span modality dropout'})
    modality_dropout_span_max: int = field(default=30, metadata={'help': 'maximum span length in frames for span modality dropout'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})
    apply_mask: bool = field(default=False, metadata={'help': 'apply feature masking during fine-tuning'})
    mask_prob: float = field(default=0.0, metadata={'help': 'feature time masking probability'})
    mask_length: int = field(default=10, metadata={'help': 'feature time mask length'})
    mask_channel_prob: float = field(default=0.0, metadata={'help': 'feature channel masking probability'})
    mask_channel_length: int = field(default=64, metadata={'help': 'feature channel mask length'})


class EncoderBackbone(BaseFairseqModel):
    def __init__(
        self,
        cfg,
        task_cfg,
        dictionaries,
        **kwargs
    ) -> None:
        super().__init__()
        logger.info(f"EncoderBackbone Config: {cfg}")

        # build encoder for different modalities
        self.feature_extractor_audio = AudioEncoder(cfg)
        self.feature_extractor_video = ResEncoder(cfg)

        # build post projection for modality fusion
        self.modality_fuse = cfg.modality_fuse
        self.modality_dropout = cfg.modality_dropout
        self.audio_dropout = cfg.audio_dropout
        self.modality_dropout_mode = getattr(cfg, "modality_dropout_mode", "sample")
        self.modality_dropout_span_min = int(getattr(cfg, "modality_dropout_span_min", 10) or 10)
        self.modality_dropout_span_max = int(getattr(cfg, "modality_dropout_span_max", 30) or 30)
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        else:
            raise ValueError(f"Unsupported modality_fuse={self.modality_fuse}")
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        # build fusion encoder
        self.encoder = TransformerEncoder(cfg)

        # # modules below are used for pre-training, we not need them during fine-tuning
        self.layer_norm = nn.LayerNorm(self.embed)

        # final_dim = (
        #     cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        # )

        # self.mask_emb = nn.Parameter(
        #     torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        # )

        # self.target_glu = None
        # if cfg.target_glu:
        #     self.target_glu = nn.Sequential(
        #         nn.Linear(final_dim, final_dim * 2), nn.GLU()
        #     )

        # self.untie_final_proj = cfg.untie_final_proj
        # if self.untie_final_proj:
        #     self.final_proj = nn.Linear(
        #         cfg.encoder_embed_dim, final_dim * len(dictionaries)
        #     )
        # else:
        #     self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # if any([d is None for d in dictionaries]):
        #     logger.info(
        #         "cannot find dictionary. assume will be used for fine-tuning"
        #     )
        # else:
        #     self.num_classes = [len(d) for d in dictionaries]
        #     self.label_embs_concat = nn.Parameter(
        #         torch.FloatTensor(sum(self.num_classes), final_dim)
        #     )
        #     nn.init.uniform_(self.label_embs_concat)
        
        # self.feat2tar_ratio = cfg.label_rate / task_cfg.sample_rate
        # self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        # self.mask_selection = cfg.mask_selection
        # self.mask_other = cfg.mask_other
        # self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        # self.no_mask_overlap = cfg.no_mask_overlap
        # self.mask_min_space = cfg.mask_min_space

        # self.mask_channel_prob = cfg.mask_channel_prob
        # self.mask_channel_selection = cfg.mask_channel_selection
        # self.mask_channel_other = cfg.mask_channel_other
        # self.mask_channel_length = cfg.mask_channel_length
        # self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        # self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.apply_mask = bool(getattr(cfg, "apply_mask", False))
        self.mask_prob = float(getattr(cfg, "mask_prob", 0.0) or 0.0)
        self.mask_length = int(getattr(cfg, "mask_length", 10) or 10)
        self.mask_channel_prob = float(getattr(cfg, "mask_channel_prob", 0.0) or 0.0)
        self.mask_channel_length = int(getattr(cfg, "mask_channel_length", 64) or 64)
        self.mask_selection = getattr(cfg, "mask_selection", "static")
        self.mask_other = float(getattr(cfg, "mask_other", 0.0) or 0.0)
        self.no_mask_overlap = bool(getattr(cfg, "no_mask_overlap", False))
        self.mask_min_space = int(getattr(cfg, "mask_min_space", 1) or 1)
        self.mask_channel_selection = getattr(cfg, "mask_channel_selection", "static")
        self.mask_channel_other = float(getattr(cfg, "mask_channel_other", 0.0) or 0.0)
        self.no_mask_channel_overlap = bool(getattr(cfg, "no_mask_channel_overlap", False))
        self.mask_channel_min_space = int(getattr(cfg, "mask_channel_min_space", 1) or 1)
        # self.logit_temp = cfg.logit_temp
        # self.skip_masked = cfg.skip_masked
        # self.skip_nomask = cfg.skip_nomask
        # self.sim_type = cfg.sim_type
        # self.selection_type = cfg.selection_type
        # self.masking_type = cfg.masking_type

    def _sample_modality_dropout_masks(self, bsz, tsz, device):
        drop_prob = float(self.modality_dropout)
        if not self.training or drop_prob <= 0:
            return None, None

        selected = torch.rand(bsz, device=device) < drop_prob
        if not selected.any():
            return None, None

        drop_audio = torch.rand(bsz, device=device) < float(self.audio_dropout)
        audio_selected = selected & drop_audio
        video_selected = selected & ~drop_audio

        if self.modality_dropout_mode not in ("sample", "span"):
            raise ValueError(f"Unsupported modality_dropout_mode={self.modality_dropout_mode}")

        audio_mask = torch.zeros(bsz, tsz, dtype=torch.bool, device=device)
        video_mask = torch.zeros(bsz, tsz, dtype=torch.bool, device=device)

        if self.modality_dropout_mode == "sample":
            audio_mask[audio_selected] = True
            video_mask[video_selected] = True
            return audio_mask, video_mask

        min_len = max(1, min(int(self.modality_dropout_span_min), tsz))
        max_len = max(min_len, min(int(self.modality_dropout_span_max), tsz))
        selected_indices = selected.nonzero(as_tuple=False).flatten()
        for batch_idx in selected_indices.tolist():
            span_len = int(torch.randint(min_len, max_len + 1, (1,), device=device).item())
            max_start = max(tsz - span_len, 0)
            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            end = start + span_len
            if bool(drop_audio[batch_idx]):
                audio_mask[batch_idx, start:end] = True
            else:
                video_mask[batch_idx, start:end] = True
        return audio_mask, video_mask
    
    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
    
    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask
    
    def apply_modality_dropout(self, features_audio, features_video):
        drop_prob = float(self.modality_dropout)
        if (
            not self.training
            or drop_prob <= 0
            or features_audio is None
            or features_video is None
        ):
            return features_audio, features_video

        ################################################################################
        # TODO:                                                                        #
        #  Apply modality dropout during audio-video training. This function is only   #
        #  called with both audio and video features available. It should return the   #
        #  possibly modified audio/video features with the same shapes as the inputs.  #
        #                                                                              #
        #  Inputs and outputs:                                                         #
        #  - features_audio and features_video both have shape [B, C, T].              #
        #  - self.modality_dropout is the probability of selecting a sample for        #
        #    modality dropout.                                                         #
        #  - self.audio_dropout decides which modality to drop after a sample has      #
        #    been selected: drop audio with this probability, otherwise drop video.    #
        #  - return tensors must keep shape [B, C, T].                                 #
        #                                                                              #
        #  Required behavior:                                                          #
        #  1. Decide independently for each batch item whether modality dropout should #
        #     be applied, according to self.modality_dropout.                          #
        #  2. For every selected item, decide whether to corrupt audio or video using  #
        #     self.audio_dropout. Non-selected items must remain unchanged.            #
        #  3. In "span" mode, corrupt only one contiguous time span in the selected    #
        #     modality. The span length should be sampled between                      #
        #     self.modality_dropout_span_min and self.modality_dropout_span_max, after #
        #     clipping that range so it is valid for the current sequence length T.    #
        #  4. In "sample" mode, corrupt the selected modality for the whole sample,    #
        #     across all channels and all time steps.                                  #
        #                                                                              #
        #  You may use tensor.masked_fill to zero whole-sample masks in sample mode.   #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        bsz, _, tsz = features_audio.shape
        audio_mask, video_mask = self._sample_modality_dropout_masks(
            bsz, tsz, features_audio.device
        )
        if audio_mask is None:
            return features_audio, features_video

        features_audio = features_audio.clone()
        features_video = features_video.clone()
        features_audio = features_audio.masked_fill(audio_mask.unsqueeze(1), 0.0)
        features_video = features_video.masked_fill(video_mask.unsqueeze(1), 0.0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return features_audio, features_video

    def apply_fused_modality_dropout(self, features):
        if self.modality_fuse != "concat" or self.modality_dropout <= 0:
            return features

        bsz, tsz, dim = features.shape
        split = self.encoder_embed_dim
        if dim != split * 2:
            return features

        audio_mask, video_mask = self._sample_modality_dropout_masks(
            bsz, tsz, features.device
        )
        if audio_mask is None:
            return features

        features = features.clone()
        features[:, :, :split] = features[:, :, :split].masked_fill(
            audio_mask.unsqueeze(-1), 0.0
        )
        features[:, :, split:] = features[:, :, split:].masked_fill(
            video_mask.unsqueeze(-1), 0.0
        )
        return features

    def fuse_modalities(self, features_audio, features_video, padding_mask=None):
        if self.modality_fuse == 'concat':
            return torch.cat([features_audio, features_video], dim=1)
        if self.modality_fuse == 'add':
            return features_audio + features_video
        raise ValueError(f"Unsupported modality_fuse={self.modality_fuse}")

    def extract_finetune(self, source, padding_mask=None, mask=False):
        src_audio, src_video = source['audio'], source['video']
        src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        if src_audio is not None and src_video is not None and self.modality_fuse != "concat":
            features_audio, features_video = self.apply_modality_dropout(features_audio, features_video)
        features = self.fuse_modalities(features_audio, features_video, padding_mask)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if src_audio is not None and src_video is not None and self.modality_fuse == "concat":
            features = self.apply_fused_modality_dropout(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if mask and self.apply_mask and self.training:
            x = self.apply_feature_mask(features, padding_mask)
        else:
            x = features

        # feature: (B, T, D), float
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        return x, padding_mask
    
    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def apply_feature_mask(self, features, padding_mask):
        if self.mask_prob <= 0 and self.mask_channel_prob <= 0:
            return features

        ################################################################################
        # TODO:                                                                        #
        #  Apply feature masking during fine-tuning.                                   #
        #  This function receives fused features after projection/dropout and should   #
        #  return features with the same shape.                                        #
        #                                                                              #
        #  Inputs and outputs:                                                         #
        #  - features has shape [B, T, C].                                             #
        #  - padding_mask has shape [B, T], where True marks padding positions.        #
        #  - return tensor must keep shape [B, T, C].                                  #
        #                                                                              #
        #  Required behavior:                                                          #
        #  1. If self.mask_prob > 0, apply time masking. Time masking selects spans    #
        #     along the T dimension and zeros all feature channels at the selected     #
        #     time steps. It should respect padding_mask, so padding positions are not #
        #     treated as normal frames when masks are sampled.                         #
        #  2. If self.mask_channel_prob > 0, apply channel masking. Channel masking    #
        #     selects spans along the C dimension and zeros the selected channels for  #
        #     every time step. It does not need padding_mask.                          #
        #  3. Time masking and channel masking are independent; either one, both, or   #
        #     neither may be active depending on the configured probabilities.         #
        #                                                                              #
        #  You can use compute_mask_indices to sample both time masks and channel      #
        #  masks. Its first return value is the boolean mask to apply.                 #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        bsz, tsz, channels = features.shape
        x = features.clone()

        if self.mask_prob > 0:
            time_mask, _, _, _ = compute_mask_indices(
                (bsz, tsz),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            time_mask = torch.from_numpy(time_mask).to(x.device)
            x = x.masked_fill(time_mask.unsqueeze(-1), 0.0)

        if self.mask_channel_prob > 0:
            channel_mask, _, _, _ = compute_mask_indices(
                (bsz, channels),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            channel_mask = torch.from_numpy(channel_mask).to(x.device)
            x = x.masked_fill(channel_mask.unsqueeze(1), 0.0)

        features = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return features

    # # methods below are used for pre-training, we not need them during fine-tuning
    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""

    #     super().upgrade_state_dict_named(state_dict, name)
    #     return state_dict

    # def apply_input_mask(self, x, padding_mask, target_list):
    #     B, C, T = x.shape[:3]
    #     is_audio = True if len(x.shape) == 3 else False
    #     if is_audio:
    #         mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
    #     else:
    #         mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
    #     if mask_prob > 0:

    #         mask_indices, starts, ends, batch_indexes = compute_mask_indices(
    #             (B, T),
    #             padding_mask,
    #             mask_prob,
    #             mask_length,
    #             self.mask_selection,
    #             self.mask_other,
    #             min_masks=2,
    #             no_overlap=self.no_mask_overlap,
    #             min_space=self.mask_min_space,
    #         )
    #         mask_indices_np = mask_indices
    #         mask_indices = torch.from_numpy(mask_indices).to(x.device)
    #         x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
    #         if B == 1:
    #             x[mask_indices] = 0
    #         elif is_audio:
    #             x[mask_indices] = self.mask_emb
    #         elif self.selection_type == 'same_other_seq':
    #             perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
    #             x_perm = x[perm]
    #             x[mask_indices] = x_perm[mask_indices]
    #         elif self.selection_type == 'same_seq':
    #             batch_indexes_, other_indexes = [], []
    #             for batch_index, start, end in zip(batch_indexes, starts, ends):
    #                 length = end-start
    #                 other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
    #                 if len(other_start) > 0:
    #                     other_start = np.random.choice(other_start, size=1)
    #                 else:
    #                     other_start = 0
    #                 other_end = other_start + length
    #                 other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
    #                 batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
    #             batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
    #             x[mask_indices] = x[batch_indexes, other_indexes]

    #         x = x.transpose(1, 2).contiguous()
    #     else:
    #         mask_indices = None

    #     if self.mask_channel_prob > 0:
    #         logger.info(f"No mask channel prob for input masking")
    #     return x, mask_indices

    # def apply_feature_mask(self, x, padding_mask, target_list):
    #     B, T, C = x.shape
    #     assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
    #     mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
    #     if mask_prob > 0:
    #         mask_indices, _, _, _ = compute_mask_indices(
    #             (B, T),
    #             padding_mask,
    #             mask_prob,
    #             mask_length,
    #             self.mask_selection,
    #             self.mask_other,
    #             min_masks=2,
    #             no_overlap=self.no_mask_overlap,
    #             min_space=self.mask_min_space,
    #         )
    #         mask_indices = torch.from_numpy(mask_indices).to(x.device)
    #         x[mask_indices] = self.mask_emb
    #     else:
    #         mask_indices = None

    #     if self.mask_channel_prob > 0:
    #         mask_channel_indices, _, _, _ = compute_mask_indices(
    #             (B, C),
    #             None,
    #             self.mask_channel_prob,
    #             self.mask_channel_length,
    #             self.mask_channel_selection,
    #             self.mask_channel_other,
    #             no_overlap=self.no_mask_channel_overlap,
    #             min_space=self.mask_channel_min_space,
    #         )
    #         mask_channel_indices = (
    #             torch.from_numpy(mask_channel_indices)
    #             .to(x.device)
    #             .unsqueeze(1)
    #             .expand(-1, T, -1)
    #         )
    #         x[mask_channel_indices] = 0

    #     return x, mask_indices

    # def forward_targets(
    #         self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # Trim features to ensure labels exist and then get aligned labels
    #     feat_tsz = features.size(2)
    #     targ_tsz = min([t.size(1) for t in target_list])
    #     if self.feat2tar_ratio * feat_tsz > targ_tsz:
    #         feat_tsz = int(targ_tsz / self.feat2tar_ratio)
    #         features = features[..., :feat_tsz]
    #         if mask_indices is not None:
    #             mask_indices = mask_indices[..., :feat_tsz]
    #     target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
    #     target_list = [t[:, target_inds.long()] for t in target_list]
    #     return features, mask_indices, target_list

    # def compute_logits(self, feats, emb_mat):
    #     # feats: [B, T, F], emb_mat: [V, F]
    #     if self.sim_type == 'dot':
    #         logits = torch.matmul(feats, emb_mat.transpose(0, 1))
    #     elif self.sim_type == 'cosine':
    #         batch_size, timesteps, emb_dim = feats.size()
    #         feats_ = feats.view(-1, emb_dim)
    #         nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
    #         denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
    #         logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
    #     else:
    #         raise NotImplementedError
    #     logits = logits / self.logit_temp
    #     return logits

    # def forward(
    #     self,
    #     source: torch.Tensor,
    #     target_list: Optional[List[torch.Tensor]] = None,
    #     padding_mask: Optional[torch.Tensor] = None,
    #     mask: bool = True,
    #     features_only: bool = False,
    #     output_layer: Optional[int] = None
    # ) -> Dict[str, torch.Tensor]:
    #     """output layer is 1-based"""
    #     src_audio, src_video = source['audio'], source['video']
    #     if mask and self.masking_type == 'input':
    #         src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
    #         src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
    #         mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
    #     else:
    #         src_audio, src_video, mask_indices = src_audio, src_video, None

    #     features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
    #     features_video = self.forward_features(src_video, modality='video')
    #     if self.modality_fuse == 'concat':
    #         features = torch.cat([features_audio, features_video], dim=1)
    #     elif self.modality_fuse == 'add':
    #         features = features_audio + features_video
    #     if target_list is not None:
    #         features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

    #     features_pen = features.float().pow(2).mean()

    #     features = features.transpose(1, 2)
    #     features = self.layer_norm(features)

    #     if padding_mask is not None:
    #         padding_mask = self.forward_padding_mask(features, padding_mask)

    #     if self.post_extract_proj is not None:
    #         features = self.post_extract_proj(features)

    #     features = self.dropout_input(features)
    #     if self.masking_type == 'feature' and mask:
    #         x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
    #     else:
    #         x = features

    #     # feature: (B, T, D), float
    #     # target: (B, T), long
    #     # x: (B, T, D), float
    #     # padding_mask: (B, T), bool
    #     # mask_indices: (B, T), bool
    #     x, _ = self.encoder(
    #         x,
    #         padding_mask=padding_mask,
    #         layer=None if output_layer is None else output_layer - 1
    #     )

    #     if features_only:
    #         return {"x": x, "padding_mask": padding_mask, "features": features}

    #     label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
    #     proj_x = self.final_proj(x)
    #     if self.untie_final_proj:
    #         proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
    #     else:
    #         proj_x_list = [proj_x for _ in self.num_classes]
    #     logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, self.num_classes)] # [[B*T, V]]
    #     mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices, ~padding_mask).view(-1) # [B*T]
    #     logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
    #     target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [target.view(-1)[unmask].long() for target in target_list]
    #     result = {
    #         "logit_m_list": logit_m_list,
    #         "logit_u_list": logit_u_list,
    #         "target_m_list": target_m_list,
    #         "target_u_list": target_u_list,
    #         "padding_mask": padding_mask,
    #         "features_pen": features_pen,
    #     }
    #     return result

    # def extract_features(
    #     self,
    #     source: torch.Tensor,
    #     padding_mask: Optional[torch.Tensor] = None,
    #     mask: bool = False,
    #     ret_conv: bool = False,
    #     output_layer: Optional[int] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     res = self.forward(
    #         source,
    #         padding_mask=padding_mask,
    #         mask=mask,
    #         features_only=True,
    #         output_layer=output_layer,
    #     )
    #     feature = res["features"] if ret_conv else res["x"]
    #     return feature, res["padding_mask"]


    # def get_extra_losses(self, net_output):
    #     extra_losses = []
    #     names = []
    #     if "features_pen" in net_output:
    #         extra_losses.append(net_output["features_pen"])
    #         names.append("features_pen")

    #     return extra_losses, names

    # def get_logits(self, net_output, is_masked=True):
    #     raise NotImplementedError

    # def get_targets(self, net_output, is_masked=True):
    #     raise NotImplementedError

    # def compute_nce(self, x, pos, negs):
    #     neg_is_pos = (pos == negs).all(-1)
    #     pos = pos.unsqueeze(0)
    #     targets = torch.cat([pos, negs], dim=0)

    #     logits = torch.cosine_similarity(
    #         x.float(), targets.float(), dim=-1
    #     ).type_as(x)
    #     logits /= self.logit_temp
    #     if neg_is_pos.any():
    #         logits[1:][neg_is_pos] = float("-inf")
    #     logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    #     return logits
