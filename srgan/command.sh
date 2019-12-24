### pretrain generator (SR module)
python train.py --name srresnet-mse --content-loss mse --train-dir data


### start gan
#python train.py --name srgan-mse --use-gan --content-loss mse --train-dir data --load-gen results/srresnet-mse/(appropriate weight name) --gpu 0
