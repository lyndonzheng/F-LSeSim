set -ex
python test.py \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/horse2zebra \
--checkpoints_dir ./checkpoints \
--name horse2zebra \
--model sc \
--num_test 0