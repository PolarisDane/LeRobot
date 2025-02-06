from dataclasses import dataclass, field
from torch import optim


@dataclass
class ATMConfig:

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.view1": [3, 480, 640],
            "observation.state": [14],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [14],
        }
    )
    # Training
    lr: float = 5e-4
    batch_size: int = 128
    mix_precision: bool = False
    num_workers: int = 8
    val_freq: int = 5
    save_freq: int = 10
    clip_grad: float = 100.0
    epochs: int = 101
    seed: int = 0
    dry: bool = False

    img_size: int = 128
    frame_stack: int = 10
    num_track_ts: int = 16
    num_track_ids: int = 32
    extra_state_keys: list[str] = field(
        default_factory=lambda: ["joint_states", "gripper_states"]
    )
    aug_prob: float = 0.9

    model_name: str = "BCViLTPolicy"
    model_cfg: dict = field(
        default_factory=lambda: {
            "load_path": None,
            "obs_cfg": {
                "obs_shapes": {"rgb": [3, 128, 128], "tracks": [16, 32, 2]},
                "img_mean": [0.0, 0.0, 0.0],
                "img_std": [1.0, 1.0, 1.0],
                "num_views": 2,
                "extra_states": ["joint_states", "gripper_states"],
                "max_seq_len": 10,
            },
            "img_encoder_cfg": {
                "network_name": "PatchEncoder",
                "patch_size": [8, 8],
                "embed_size": 128,
                "no_patch_embed_bias": False,
            },
            "language_encoder_cfg": {
                "network_name": "MLPEncoder",
                "input_size": 768,
                "hidden_size": 128,
                "num_layers": 1,
            },
            "extra_state_encoder_cfg": {
                "extra_num_layers": 0,
                "extra_hidden_size": 128,
            },
            "track_cfg": {
                "track_fn": None,
                "policy_track_patch_size": 16,
                "use_zero_track": False,
            },
            "spatial_transformer_cfg": {
                "num_layers": 7,
                "num_heads": 8,
                "head_output_size": 120,
                "mlp_hidden_size": 256,
                "dropout": 0.1,
                "spatial_downsample": True,
                "spatial_downsample_embed_size": 64,
                "use_language_token": False,
            },
            "temporal_transformer_cfg": {
                "num_layers": 4,
                "num_heads": 6,
                "head_output_size": 64,
                "mlp_hidden_size": 256,
                "dropout": 0.1,
                "use_language_token": False,
            },
            "policy_head_cfg": {
                "network_name": "DeterministicHead",
                "output_size": [
                    7,
                ],
                "hidden_size": 1024,
                "num_layers": 2,
                "loss_coef": 1.0,
                "action_squash": False,
            },
        }
    )
    dataset_cfg: dict = field(
        default_factory=lambda: {
            "img_size": 128,
            "frame_stack": 10,
            "num_track_ts": 16,
            "num_track_ids": 32,
            "track_obs_fs": 1,
            "augment_track": False,
            "extra_state_keys": ["joint_states", "gripper_states"],
            "cache_all": True,
            "cache_image": True,
        }
    )
