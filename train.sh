CUDA_VISIBLE_DEVICES=3 python train.py --dataroot /data_zfs/zhoubingcheng/gan_datasets/gan_aligned_crop_data --name document_pix2pix_best --model pix2pix \
 --netG unet_256 --direction AtoB --lambda_L1 100 --save_latest_freq 10000 --dataset_mode aligned --norm batch --num_threads 8

CUDA_VISIBLE_DEVICES=5 python train.py --dataroot /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_crop_signet_data_1 \
  --name document_pix2pix_best1 --model pix2pix --netG unet_256 --direction AtoB \
   --save_latest_freq 20000 --dataset_mode aligned --norm batch --num_threads 16 \
   --n_epochs 20 --n_epochs_decay 10 --lr 0.001 --gan_mode lsgan

