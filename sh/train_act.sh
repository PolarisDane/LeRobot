DATASET="t20003"
DATE="2025_2_20"
DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${DATASET} \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/act_${DATE}_${DATASET} \
  hydra.job.name=act_${DATE}_${DATASET} \
  device=cuda \
  wandb.enable=true \
  resume=false








