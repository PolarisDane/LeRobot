from dataclasses import dataclass, field


@dataclass
class SNNConfig:


    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.view_front": [3, 480, 640],
            "observation.state": [6],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [6],
        }
    )
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.view_front": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "action": "min_max",
        }
    )

    num_action: int = 20
    input_dim : int = 6
    img_dim : int = 64
    state_dim : int = 6
    hidden_dim : int = 256
    action_dim : int = 6
    dropout : float= 0.1

    device : str = 'cuda'