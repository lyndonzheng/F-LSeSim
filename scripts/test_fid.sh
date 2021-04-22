set -ex
python test_fid.py \
--dataroot /media/lyndon/c6f4bbbd-8d47-4dcb-b0db-d788fe2b25571/dataset/image_translation/horse2zebra \
--name horse2zebra_F47_cosT4tgt_p256_oriCUT_w10_s64 \
--model sc \
--num_test 0