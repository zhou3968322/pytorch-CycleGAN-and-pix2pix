CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /data_zfs/zhoubingcheng/gan_datasets/gan_aligned_crop_data --name document_pix2pix_best --model pix2pix \
 --netG unet_256 --direction AtoB --lambda_L1 100 --save_latest_freq 10000 --dataset_mode aligned --norm batch --num_threads 8

CUDA_VISIBLE_DEVICES=7 python train.py --dataroot /data_zfs/zhoubingcheng/gan_datasets/gan_aligned_crop_data \
 --name document_cyclegan_best --model cycle_gan --lambda_A 10.0 --lambda_B 5.0 \
 --save_latest_freq 10000 --preproces resize --dataset_mode aligned_resize --load_size 512 \
  --display_port 8098 --num_threads 8
