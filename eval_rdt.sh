   python eval_rdt.py \
      --lang_embeddings_path="/home/robot/lerobot/lerobot/common/policies/rdt/outs/koch_sponge.pt" \
      --pretrained_model_name_or_path="/home/robot/lerobot/lerobot/common/policies/rdt/checkpoints/rdt-finetune-170m-bs128/checkpoint-40000" --ctrl_freq=30 \
      --config_path="/home/robot/lerobot/lerobot/common/policies/rdt/configs/base.yaml"