#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from spikingjelly.activation_based.neuron import LIFNode, IFNode
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18
from spikingjelly.activation_based.encoding import PoissonEncoder
from spikingjelly.activation_based.layer import Dropout, BatchNorm1d, BatchNorm2d

from lerobot.common.policies.spikeact.configuration_spikeact import SpikeACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.spikeact.model.models import (
    MS_MultiheadAttention,
    mem_update,
)


class SpikeACTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "act"],
):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    name = "spikeact"

    def __init__(
        self,
        config: SpikeACTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = SpikeACTConfig()
        self.config: SpikeACTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = SpikeACT(config)

        self.expected_image_keys = [
            k for k in config.input_shapes if k.startswith("observation.image")
        ]

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = SpikeACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )
        self.T = 4
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.expected_image_keys], dim=-4
            )
        batch["observation.images"] = batch["observation.images"].repeat(
            self.T, 1, 1, 1, 1, 1
        )
        batch["observation.state"] = batch["observation.state"].repeat(self.T, 1, 1)
        # batch["action"] = batch["action"].repeat(self.T, 1, 1, 1)
        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch).mean(dim=0)[
                0
            ]  
            # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0].mean(dim=0)[:, : self.config.n_action_steps]
            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.expected_image_keys], dim=-4
            )
        batch = self.normalize_targets(batch)
        batch["observation.images"] = batch["observation.images"].repeat(
            self.T, 1, 1, 1, 1, 1
        )
        batch["observation.state"] = batch["observation.state"].repeat(self.T, 1, 1)
        batch["action"] = batch["action"].repeat(self.T, 1, 1, 1)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # batch["action_is_pad"] = batch["action_is_pad"].repeat(self.T,1,1)

        l1_loss = (
            F.l1_loss(batch["action"][0], actions_hat.mean(dim=0), reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            mu_hat = mu_hat.mean(dim=0)
            log_sigma_x2_hat = log_sigma_x2_hat.mean(dim=0)
            mean_kld = (
                (
                    -0.5
                    * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())
                )
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss_dict["loss"] = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss_dict["loss"] = l1_loss

        functional.reset_net(self.model)
        return loss_dict


class SpikeACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(
            -temporal_ensemble_coeff * torch.arange(chunk_size)
        )
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(
            device=actions.device
        )
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1),
                dtype=torch.long,
                device=self.ensembled_actions.device,
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[
                self.ensembled_actions_count - 1
            ]
            self.ensembled_actions += (
                actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            )
            self.ensembled_actions /= self.ensemble_weights_cumsum[
                self.ensembled_actions_count
            ]
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size
            )
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1
            )
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones_like(self.ensembled_actions_count[-1:]),
                ]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class SpikeACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: SpikeACTConfig):
        super().__init__()
        self.config = config
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.T = 4
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(
            k.startswith("observation.image") for k in config.input_shapes
        )
        self.use_env_state = "observation.environment_state" in config.input_shapes
        if self.config.use_vae:
            self.vae_encoder = SpikeACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.use_robot_state:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.input_shapes["observation.state"][0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                config.output_shapes["action"][0], config.dim_model
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(
                config.dim_model, config.latent_dim * 2
            )
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.use_robot_state:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(
                    num_input_token_encoder, config.dim_model
                )
                .unsqueeze(0)
                .unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.use_images:
            # backbone_model = getattr(torchvision.models, config.vision_backbone)(
            #     replace_stride_with_dilation=[
            #         False,
            #         False,
            #         config.replace_final_stride_with_dilation,
            #     ],
            #     weights=config.pretrained_backbone_weights,
            #     norm_layer=FrozenBatchNorm2d,
            # )
            backbone_model = spiking_resnet18(
                pretrained=True,
                spiking_neuron=mem_update,
                # step_mode="m",
                # tau=2.0,
                # surrogate_function=surrogate.ATan(),
                # detach_reset=True,
                # backend="cupy",
            )
            functional.set_backend(backbone_model, "cupy")
            functional.set_step_mode(backbone_model, "m")
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(
                backbone_model, return_layers={"layer4": "feature_map"}
            )
            print(self.backbone)
        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = SpikeACTEncoder(config)
        self.decoder = SpikeACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.use_robot_state:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.input_shapes["observation.state"][0], config.dim_model
            )
            self.encoder_robot_state_input_norm = BatchNorm1d(
                config.dim_model, step_mode="m"
            )
        if self.use_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.input_shapes["observation.environment_state"][0],
                config.dim_model,
            )
            self.encoder_env_state_input_norm = BatchNorm1d(
                config.dim_model, step_mode="m"
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        self.encoder_latent_input_norm = BatchNorm1d(config.dim_model, step_mode="m")
        if self.use_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            self.encoder_img_feat_input_norm = BatchNorm2d(
                config.dim_model, step_mode="m"
            )

        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.use_robot_state:
            n_1d_tokens += 1
        if self.use_env_state:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = SpikeACTSinusoidalPositionEmbedding2d(
                config.dim_model // 2
            )

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.lif_head = mem_update()
        self.action_head = nn.Linear(
            config.dim_model, config.output_shapes["action"][0]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.

            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."

        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[1]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight,
                "1 d -> t b 1 d",
                b=batch_size,
                t=self.T,
            )  # (T, B, 1, D)
            if self.use_robot_state:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    batch["observation.state"]
                )
                robot_state_embed = robot_state_embed.unsqueeze(2)  # (T, B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(
                batch["action"]
            )  # (T, B, S, D)

            if self.use_robot_state:
                vae_encoder_input = [
                    cls_embed,
                    robot_state_embed,
                    action_embed,
                ]  # (T, B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=2)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, 1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.use_robot_state else 1),
                False,
                device=batch["observation.state"].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)
            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input,  # (T, B, S+2, D)
                pos_embed=pos_embed,  # (T, B, S+2, D)
                key_padding_mask=key_padding_mask,  # (B, S+1)
            )[
                :, :, 0
            ]  # select the class token, with shape (T, B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, :, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, :, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros(
                [self.T, batch_size, self.config.latent_dim], dtype=torch.float32
            ).to(batch["observation.state"].device)

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [
            self.encoder_latent_input_proj(latent_sample)
        ]  # [(T,B,dim)]
        encoder_in_pos_embed = list(
            self.encoder_1d_feature_pos_embed.weight.unsqueeze(1).unsqueeze(
                1
            )  # [1, 1, dim] * token
        )
        # Robot state token.
        if self.use_robot_state:
            encoder_in_tokens.append(
                self.encoder_robot_state_input_norm(
                    self.encoder_robot_state_input_proj(batch["observation.state"])
                )  # (T, B, dim)
            )
        # Environment state token.
        if self.use_env_state:
            encoder_in_tokens.append(
                self.encoder_env_state_input_norm(
                    self.encoder_env_state_input_proj(
                        batch["observation.environment_state"]
                    )
                )  # (T, B, dim)
            )

        # Camera observation features and positional embeddings.
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []

            for cam_index in range(batch["observation.images"].shape[-4]):
                cam_features = self.backbone(
                    batch["observation.images"][:, :, cam_index]
                )[
                    "feature_map"
                ]  # (T, B, C, H, W)
                # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
                # buffer
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )  # (1, 1, C, H, W)
                cam_features = self.encoder_img_feat_input_proj(
                    cam_features.flatten(0, 1)
                ).unflatten(
                    0, (self.T, batch_size)
                )  # (T, B, C, h, w)
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)
            # Concatenate camera observation feature maps and positional embeddings along the width dimension,
            # and move to (sequence, batch, dim).
            all_cam_features = torch.cat(all_cam_features, axis=-1)
            all_cam_features = self.encoder_img_feat_input_norm(all_cam_features)
            encoder_in_tokens.extend(
                einops.rearrange(all_cam_features, "t b c h w -> (h w) t b c")
            )
            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
            encoder_in_pos_embed.extend(
                einops.rearrange(all_cam_pos_embeds, "t b c h w -> (h w) t b c")
            )

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(
            encoder_in_tokens, axis=2
        )  # (T, B, (H*W)+token, C=dim)
        encoder_in_pos_embed = torch.stack(
            encoder_in_pos_embed, axis=2
        )  # (1, 1, (H*W)+token, C=dim)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.T, batch_size, self.config.chunk_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )  # (T, B, L, dim)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(0).unsqueeze(0),
        )

        # Move back to (T, B, S, C).
        # decoder_out = decoder_out.transpose(0, 1)
        decoder_out = self.lif_head(decoder_out)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class SpikeACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: SpikeACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = (
            config.n_vae_encoder_layers
            if self.is_vae_encoder
            else config.n_encoder_layers
        )
        self.layers = nn.ModuleList(
            [SpikeACTEncoderLayer(config) for _ in range(num_layers)]
        )
        self.norm = (
            BatchNorm1d(config.dim_model, step_mode="m")
            if config.pre_norm
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        # x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        return x


class SpikeACTEncoderLayer(nn.Module):
    def __init__(self, config: SpikeACTConfig):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(
        #     config.dim_model, config.n_heads, dropout=config.dropout
        # )
        self.self_attn = MS_MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = Dropout(config.dropout, step_mode="m")
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = BatchNorm1d(config.dim_feedforward, step_mode="m")
        self.norm2 = BatchNorm1d(config.dim_model, step_mode="m")
        self.dropout1 = Dropout(config.dropout, step_mode="m")
        self.dropout2 = Dropout(config.dropout, step_mode="m")

        # self.activation = get_activation_fn(config.feedforward_activation)
        # self.activation = LIFNode(
        #     tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        # )
        self.activation = mem_update()
        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.pre_norm = config.pre_norm

    def forward(
        self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        skip = x
        # if self.pre_norm:
        # x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = skip + self.dropout1(x)

        skip = x
        x = self.lif1(x)
        x = self.linear1(x)
        x = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.lif2(x)
        # dropout
        x = self.linear2(x)
        x = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = skip + x
        # x = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        return x


class SpikeACTDecoder(nn.Module):
    def __init__(self, config: SpikeACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList(
            [SpikeACTDecoderLayer(config) for _ in range(config.n_decoder_layers)]
        )
        self.norm = BatchNorm1d(config.dim_model, step_mode="m")

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        # if self.norm is not None:
        # x = self.norm(x)
        # x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        return x


class SpikeACTDecoderLayer(nn.Module):
    def __init__(self, config: SpikeACTConfig):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(
        #     config.dim_model, config.n_heads, dropout=config.dropout
        # )
        # self.multihead_attn = nn.MultiheadAttention(
        #     config.dim_model, config.n_heads, dropout=config.dropout
        # )
        self.self_attn = MS_MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        self.multihead_attn = MS_MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = Dropout(config.dropout, step_mode="m")
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = BatchNorm1d(config.dim_feedforward, step_mode="m")
        self.norm2 = BatchNorm1d(config.dim_model, step_mode="m")
        # self.norm3 = BatchNorm1d(config.dim_model, step_mode="m")
        self.dropout1 = Dropout(config.dropout, step_mode="m")
        self.dropout2 = Dropout(config.dropout, step_mode="m")
        self.dropout3 = Dropout(config.dropout, step_mode="m")

        # self.activation = get_activation_fn(config.feedforward_activation)
        # self.activation = LIFNode(
        #     tau=2.0, step_mode="m", detach_reset=True, backend="cupy"
        # )
        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x  # (T,B,L,dim)
        # if self.pre_norm:
        # x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(
            q, k, value=x
        )  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        # x = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )  # select just the output, not the attention weights
        x = skip + self.dropout2(x)

        skip = x
        x = self.lif1(x)
        x = self.linear1(x)
        x = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.lif2(x)
        # dropout
        x = self.linear2(x)
        x = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = skip + x
        # x = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)

        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension)
            for hid_j in range(dimension)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class SpikeACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (T, B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, 1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, 0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2
            * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
            / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack(
            (x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed_y = torch.stack(
            (y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1
        ).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)[
            None, ...
        ]  # (1, 1, C, H, W)

        return pos_embed
