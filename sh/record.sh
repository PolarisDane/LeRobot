python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root . \
  --repo-id data/t20005_limited_randomization \
  --warmup-time-s 5 \
  --episode-time-s 50 \
  --reset-time-s 50 \
  --num-episodes 66 \
  --push-to-hub 0