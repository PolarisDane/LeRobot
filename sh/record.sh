python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root . \
  --repo-id data/t20003_dual \
  --warmup-time-s 5 \
  --episode-time-s 50 \
  --reset-time-s 50 \
  --num-episodes 60 \
  --push-to-hub 0