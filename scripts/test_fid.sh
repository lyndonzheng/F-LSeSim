set -ex
python test_fid.py \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/horse2zebra \
--name horse2zebra_F47_cosT2tgt_p256_p1_oriCUT_norm_w10 \
--model sc \
--num_test 0