from dataclasses import dataclass, field


@dataclass
class RISEConfig:


    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.state": [6],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [6],
        }
    )
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "action": "mean_std",
        }
    )
    num_action: int = 20
    input_dim : int = 6
    num_obs : int = 1
    obs_feature_dim : int = 512
    action_dim : int = 10
    hidden_dim : int = 512
    nheads : int = 8
    num_encoder_layers : int = 4
    num_decoder_layers : int = 1
    dim_feedforward : int = 2048
    dropout : float= 0.1

    device : str = 'cuda'