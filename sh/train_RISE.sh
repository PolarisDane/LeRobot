DATASET="t10003"
DATE="2025_2_21_t10003"
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=$DATASET \
  policy=RISE_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/RISE_$DATE \
  hydra.job.name=RISE_$DATE \
  device=cuda \
  wandb.enable=true \
  resume=false