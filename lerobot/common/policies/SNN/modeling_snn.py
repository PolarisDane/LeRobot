import torch
from torch import Tensor, nn
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from huggingface_hub import PyTorchModelHubMixin
from collections import deque

from spikingjelly.activation_based.model.spiking_resnet import spiking_resnet18
from spikingjelly.activation_based import neuron, surrogate, functional

from lerobot.common.policies.SNN.configuration_snn import SNNConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize


class SNNPolicy(nn.Module,PyTorchModelHubMixin,):
    
    name='snn'

    def __init__(
        self,
        config: SNNConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        if config is None:
            config = SNNConfig()
        self.config: SNNConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.device=config.device
        self.encoder1 = ResnetEncoder(config.input_shapes["observation.images.view_front"],config.img_dim,True)
        
        self.net = nn.Sequential(
            nn.Linear(config.state_dim + config.img_dim , config.hidden_dim),
            neuron.LIFNode(tau=2.0,surrogate_function=surrogate.ATan()),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            neuron.LIFNode(tau=2.0,surrogate_function=surrogate.ATan()),
            nn.Linear(config.hidden_dim, config.action_dim*config.num_action),
            neuron.NonSpikingLIFNode(tau=2.0),
        )
        torch.nn.init.xavier_uniform_(self.net[0].weight)
        torch.nn.init.xavier_uniform_(self.net[2].weight)  
        torch.nn.init.xavier_uniform_(self.net[4].weight)


        functional.set_step_mode(self,'m')
        self.expected_image_keys = ["observation.images.view_front"]
        self.expected_depth_keys = ["observation.depths.view_front"]
    
        self._action_queue = deque([], maxlen=self.config.num_action)


    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:

            functional.reset_net(self)
            batch = self.normalize_inputs(batch)
            if len(self.expected_image_keys) > 0:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4) # b, v, c, h, w 

            batch_size = (
                batch["observation.images"]
                if "observation.images" in batch
                else batch["observation.environment_state"]
            ).shape[0]
            input_img = batch["observation.images"][:,0].repeat(4,1,1,1,1)
            input_state = batch["observation.state"].repeat(4,1,1)
            img_features = self.encoder1(input_img)
            act = self.net(torch.cat([input_state,img_features],dim=-1)).reshape(batch_size,self.config.num_action,-1)
            act = self.unnormalize_outputs({"action": act})["action"]
            self._action_queue.extend(act.transpose(0, 1))
        return self._action_queue.popleft()
  


    def forward(self, batch: dict[str, Tensor]):
        functional.reset_net(self)
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4) # b, v, c, h, w 

        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]
        input_img = batch["observation.images"][:,0].repeat(4,1,1,1,1)
        input_state = batch["observation.state"].repeat(4,1,1)
        img_features = self.encoder1(input_img)
        act = self.net(torch.cat([input_state,img_features],dim=-1)).reshape(batch_size,self.config.num_action,-1)
        loss = F.mse_loss(act, self.normalize_targets(batch)['action'])
        return {'loss':loss}

class ResnetEncoder(nn.Module):
    def __init__(self,
        input_shape,
        output_size,
        pretrained=False,
        freeze=False,
        remove_layer_num=4,
        no_stride=False,
        language_dim=768,
        language_fusion="none",):
        super().__init__()

        ### 1. encode input (images) using convolutional layers
        assert remove_layer_num <= 5, "[error] please only remove <=5 layers"
        layers = list(spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True).children())[
            :-remove_layer_num
        ]
        self.remove_layer_num = remove_layer_num
    
        assert (
            len(input_shape) == 3
        ), "[error] input shape of resnet should be (C, H, W)"

        in_channels = input_shape[0]
        if in_channels != 3:  # has eye_in_hand, increase channel size
            conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.resnet18_base = nn.Sequential(*layers[:4])
        self.block_1 = layers[4][0]
        self.block_2 = layers[4][1]
        self.block_3 = layers[5][0]
        self.block_4 = layers[5][1]

        self.language_fusion = language_fusion
        if language_fusion != "none":
            self.lang_proj1 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj2 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj3 = nn.Linear(language_dim, 128 * 2)
            self.lang_proj4 = nn.Linear(language_dim, 128 * 2)

        if freeze:
            if in_channels != 3:
                raise Exception(
                    "[error] cannot freeze pretrained "
                    + "resnet with the extra eye_in_hand input"
                )
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        ### 2. project the encoded input to a latent space
        x = torch.zeros(1, *input_shape)
        y = self.block_4(
            self.block_3(self.block_2(self.block_1(self.resnet18_base(x))))
        )
        output_shape = y.shape  # compute the out dim
        self.projection_layer = SpatialProjection(output_shape[1:], output_size)
        self.output_shape = self.projection_layer(y).shape

    def forward(self, x, langs=None):
        h = self.resnet18_base(x)

        h = self.block_1(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj1(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_2(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj2(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_3(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj3(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta

        h = self.block_4(h)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = h.shape
            beta, gamma = torch.split(
                self.lang_proj4(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            h = (1 + gamma) * h + beta
        T,B = h.shape[:2]
        h = h.reshape(-1,*h.shape[2:])
        h = self.projection_layer(h)
        h = h.reshape(T,B,*h.shape[1:])
        return h

    def output_shape(self, input_shape, shape_meta):
        return self.output_shape

    
class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert (
            len(input_shape) == 3
        ), "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)

class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints