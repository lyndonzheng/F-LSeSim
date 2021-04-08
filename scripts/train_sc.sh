set -ex
python ../train.py  \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/horse2zebra \
--name horse2zebra_F47_l1T2_p256_p1_oriCUT_w10 \
--model sc \
--gpu_ids 1 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7 \
--loss_mode l1 \
--gan_mode lsgan \
--display_port 8093
#\ --learned_attn \--augment