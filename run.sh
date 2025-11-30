lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30, fourcc: 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=30 \
  --dataset.num_episodes=50 \
  --dataset.repo_id=wair/so101_test_01 \
  --dataset.num_episodes=5 \
  --dataset.single_task="Grab the small cube and put it in the bowl." \
  --dataset.root="/home/wair/yangsheng/record_datasets/so101_test_01" \
  --dataset.push_to_hub=false


lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30, fourcc: 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.episode_time_s=50 \
  --dataset.reset_time_s=25 \
  --dataset.num_episodes=200 \
  --dataset.repo_id=wair/so101_grab \
  --dataset.single_task="Grab the plastic bottle and put it in the bowl.​ ." \
  --dataset.root="/home/wair/yangsheng/record_datasets/so101_grab_bottle_11171015" \
  --dataset.push_to_hub=false


lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30}}" \
    --robot.id=R07254808 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=R07254808 \
    --display_data=true

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=R07254808

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=R07254808

lerobot-train \
  --dataset.repo_id=/home/wair/yangsheng/record_datasets/so101_train_01 \
  --policy.type=groot \
  --output_dir=/home/wair/yangsheng/train/gr00t_so101_train_11142359 \
  --job_name=gr00t_so101_train_11142359 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false

lerobot-train \
  --dataset.repo_id=/home/wair/yangsheng/record_datasets/so101_train_01 \
  --policy.type=act \
  --output_dir=/home/wair/yangsheng/train/act_so101_train_11152251 \
  --job_name=act_so101_train_11152251 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30,  'fourcc': 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.single_task="Grab the small cube and put it in the bowl." \
  --policy.path=/home/wair/yangsheng/train/act_so101_train_11152251/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=wair/eval_so101_11170946_act \
  --dataset.push_to_hub=false

  lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30,  'fourcc': 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.single_task="Grab the small cube and put it in the bowl." \
  --policy.path=/home/wair/yangsheng/train/act_so101_train_11152251/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=wair/eval_so101_11171128_gr00t \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=10


  lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30,  'fourcc': 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.single_task="Grab the plastic bottle and put it in the bowl.​ ." \
  --policy.path=/home/wair/yangsheng/models/act_11171239/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=wair/eval_so101_11171239_gr00t \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=10

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30,  'fourcc': 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.single_task="Grab the plastic bottle and put it in the bowl.​ ." \
  --policy.path=/home/wair/yangsheng/models/gr00t_11171252 \
  --policy.device=cuda \
  --dataset.repo_id=wair/eval_so101_11171252_gr00t \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=10




python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30,  'fourcc': 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.single_task="Grab the plastic bottle and put it in the bowl.​ ." \
  --policy.path=/home/wair/yangsheng/models/gr00t_11171252 \
  --policy.device=cuda \
  --dataset.repo_id=wair/eval_so101_11171252_gr00t \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=10

  /home/wair/yangsheng/train/gr00t_so101_train_11162242/pretrained_model

  /home/wair/yangsheng/models/act_11171239/checkpoints/last/pretrained_model


lerobot-train \
  --dataset.repo_id=/home/wair/yangsheng/record_datasets/so101_train_01 \
  --policy.type=act \
  --output_dir=/home/wair/yangsheng/train/act_so101_train_11152251 \
  --job_name=act_so101_train_11152251 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false

# ============================================================================================================
conda activate lerobot_cyl

nohup pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple/  > pip.log 2>&1 &
nohup pip install -e ".[all]" -i  https://pypi.tuna.tsinghua.edu.cn/simple/  > pip.log 2>&1 &
nohup pip install -e ".[pi]" -i  https://pypi.tuna.tsinghua.edu.cn/simple/ > pip.log 2>&1 &

python src/lerobot/scripts/lerobot_record.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=R07254808 \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30, 'fourcc': 'MJPG'}, 'fixed': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30,  'fourcc': 'MJPG'}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=R07254808 \
  --display_data=true \
  --dataset.single_task="Grab the plastic bottle and put it in the bowl.​ ." \
  --policy.path=/home/wair/yangsheng/models/gr00t_11171252 \
  --policy.device=cuda \
  --dataset.repo_id=wair/eval_so101_11171252_gr00t \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=10