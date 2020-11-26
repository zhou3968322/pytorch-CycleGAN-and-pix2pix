# dl4
# 8097
CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_data --name experiment1 \
 --serial_batches --preprocess resize --crop_size 512 --num_threads 8 > experiment1.out  2>&1 &
# dl2
CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_aligned_data --name experiment2 \
--serial_batches --model cycle_gan --dataset_mode aligned --preprocess crop --load_size 512 --crop_size 512 \
--no_flip --num_threads 8 --continue_train --display_port 8098 > experiment2.out  2>&1 &

#test
#--dataroot /data_ssd/ocr/zhoubingcheng/gan_datasets/test_gan_data --name document_experiment --serial_batches --crop_size 256 --num_threads 8 --gpu_ids 0,1,2 --display_freq 100

CUDA_VISIBLE_DEVICES=9 python train.py --dataroot /data_ssd/ocr/zhoubingcheng/gan_datasets/gan_signet_aligned_data2 --name document_pix2pix --model pix2pix \
 --netG unet_256 --direction AtoB --lambda_L1 100 --save_latest_freq 10000 --continue_train --dataset_mode aligned --norm batch

CUDA_VISIBLE_DEVICES=6 python train.py --dataroot /data_zfs/zhoubingcheng/gan_datasets/gan_aligned_crop_data --name document_pix2pix_best --model pix2pix \
 --netG unet_256 --direction AtoB --lambda_L1 100 --save_latest_freq 10000 --dataset_mode aligned --norm batch --load_size 1024 --crop_size 1024
