set -ex
python train.py  \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/apple2orange \
--name apple2orange_F479_cosT4tgt_p256_oriCUT_w10_s64 \
--model sc \
--gpu_ids 0 \
--lambda_spatial 10 \
--lambda_gradient 0 \
--attn_layers 4,7,9 \
--loss_mode cos \
--gan_mode lsgan \
--display_port 8093 \
--patch_size 64
#\--learned_attn \
#--augment