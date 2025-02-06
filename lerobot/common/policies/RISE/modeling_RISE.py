import torch
from torch import Tensor, nn
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from huggingface_hub import PyTorchModelHubMixin
from collections import deque

import MinkowskiEngine as ME

from lerobot.common.policies.RISE.tokenizer import Sparse3DEncoder
from lerobot.common.policies.RISE.transformer import Transformer
from lerobot.common.policies.RISE.diffusion import DiffusionUNetPolicy
from lerobot.common.policies.RISE.configuration_RISE import RISEConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize


class RISEPolicy(nn.Module,PyTorchModelHubMixin,):
    
    name='RISE'

    def __init__(
        self,
        config: RISEConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        if config is None:
            config = RISEConfig()            
        self.config: RISEConfig = config
        print(config)
        self.sparse_encoder = Sparse3DEncoder(config.input_dim, config.obs_feature_dim)
        self.transformer = Transformer(config.hidden_dim, config.nheads, config.num_encoder_layers, config.num_decoder_layers, config.dim_feedforward, config.dropout)
        self.action_decoder = DiffusionUNetPolicy(config.action_dim, config.num_action, config.num_obs, config.obs_feature_dim)
        self.readout_embed = nn.Embedding(1, config.hidden_dim)

        self.device=config.device

        self.expected_image_keys = ["observation.images.view_front"]
        self.expected_depth_keys = ["observation.depths.view_front"]

        self._action_queue = deque([], maxlen=self.config.num_action)
        self.dataset_stats = dataset_stats

        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )


    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        if len(self._action_queue) == 0:
            cloud_coords = batch['input_coords_list'][0]
            cloud_feats = batch['input_feats_list'][0] 
            MEAN = torch.tensor([0.485, 0.456, 0.406])
            STD = torch.tensor([0.229, 0.224, 0.225])
            cloud_coords, cloud_feats = ME.utils.sparse_collate(cloud_coords, cloud_feats)
            cloud_feats[:,3:] = (cloud_feats[:,3:] - MEAN) / STD
            cloud = ME.SparseTensor(cloud_feats, cloud_coords,device=self.device)

            src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
            readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
            readout = readout[:, 0]
            action_pred = self.action_decoder.predict_action(readout)
            actions = self.unnormalize_outputs({"action": action_pred})["action"]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()


    def forward(self, batch: dict[str, Tensor]):
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        batch = self.normalize_targets(batch)

        # cloud_coords = torch.stack([item for b in batch['input_coords_list'] for item in b])
        # cloud_feats = torch.stack([item for b in batch['input_feats_list'] for item in b])
        cloud_coords = batch['input_coords_list'][0]
        cloud_feats = batch['input_feats_list'][0]
        MEAN = self.dataset_stats['observation.images.view_front']['mean'].squeeze().to(self.device) 
        STD = self.dataset_stats['observation.images.view_front']['std'].squeeze().to(self.device) 
        # for i in len(cloud_coords):
        #     cloud_coords[i]=cloud_coords[i].to(self.device)
        # for i in len(cloud_feats):
        #     cloud_feats[i]=cloud_feats[i].to(self.device)
        cloud_coords, cloud_feats = ME.utils.sparse_collate(cloud_coords, cloud_feats)
        cloud_feats[:,3:] = (cloud_feats[:,3:] - MEAN) / STD
        cloud = ME.SparseTensor(cloud_feats, cloud_coords,device=self.device)
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        loss = self.action_decoder.compute_loss(readout, batch["action"])
        return {'loss':loss}
    
