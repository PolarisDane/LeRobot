DATASET="t10000_test"
DATE="2025_2_6_t10000_test"
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=$DATASET \
  policy=dp_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/dp_$DATE \
  hydra.job.name=dp_$DATE\
  device=cuda \
  wandb.enable=true \
  resume=false