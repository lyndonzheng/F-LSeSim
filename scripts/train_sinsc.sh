set -ex
python train.py \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/single_image_monet_etretat \
--name image2monet_single_VGG4,7_non_norm_w4 \
--model sinsc \
--gpu_ids 0 \
--display_port 8093 \
--pool_size 0