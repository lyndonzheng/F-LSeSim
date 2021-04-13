set -ex
python train.py \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/single_image_monet_etretat \
--name image2monet \
--model sinsc \
--gpu_ids 1 \
--display_port 8093 \
--pool_size 0