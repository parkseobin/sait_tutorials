# Tutorial for generative models and Meta-Learning

This was originally produced as a tutorial of <b>generative models and Meta-Learning(MAML)</b> for Samsung Advanced Institute of Technology. There are some holes in the code to fill. See the powerpoint material for details. 


## How to run

- Download dataset

```shell
~/se_tutorials$ sh download_dataset.sh
```

- Create an anaconda environment

```shell
~/se_tutorials$ conda create -n (env name)
~/se_tutorials$ conda activate (env name)
(env name) ~/se_tutorials$ conda install --file requirements.txt
```

- Run tutorials

```shell
(env name) ~/se_tutorials/(topic name)$ sh command.sh
```

- topic names are: vae, dcgan, pix2pix, cyclegan, srgan, maml
- Edit command.sh in each topic directory for different settings. GPU number, flag changes etc..


## References


- vae: https://github.com/hwalsuklee/tensorflow-mnist-VAE
- dcgan: https://gist.github.com/NeuroWhAI/d852119473cdadce066503af4196bc1a
- pix2pix: https://github.com/yenchenlin/pix2pix-tensorflow
- cyclegan: https://github.com/xhujoy/CycleGAN-tensorflow
- srgan: https://github.com/trevor-m/tensorflow-SRGAN
- meta-learning: https://github.com/cbfinn/maml

