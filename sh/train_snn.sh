DATASET="t10000"
DATE="2024_12_27_t10000"
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${DATASET} \
  policy=snn_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/snn_$DATE \
  hydra.job.name=snn_$DATE \
  device=cuda \
  wandb.enable=true \
  resume=false