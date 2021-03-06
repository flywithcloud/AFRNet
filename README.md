# AFRNet
This is an implementation for [Zero-Shot Learning from Adversarial Feature Residual to Compact Visual Feature](https://arxiv.org/submit/3344955/view).
The code will be released soon.

# Datasets
Data could be download [here](https://pan.baidu.com/s/1Swib6P5fbAWOI_LrK3ND0Q) (g3ni).

# Evaluate
Trained models could be download [here](https://pan.baidu.com/s/1CjjJwlYim-BOiTi04WXmFA) (lzea) to evaluate the proposed method by running:

python evaluate_proto_feat.py --dataset APY --resSize 2048 --attSize 64 --nz 64 --syn_num 2000 --modeldir './APY_proto_model' --nepoch 40 --bs 128 --lr 0.0002

or:

python evaluate_proto_feat.py --dataset APY --resSize 1024 --attSize 64 --nz 64 --syn_num 2000 --modeldir './APY_proto1024_model' --nepoch 40 --bs 128 --lr 0.0002

# Train
Models could be trained by running:

python wgan_proto.py --dataset APY --resSize 2048 --attSize 64 --nz 64 --nepoch 5 --bs 512 --lr 0.00005 --modeldir './APY_proto_model' --syn_num 2000 --nepoch_cls 40 --bs_cls 128 --lr_cls 0.0002

or running code with feature selection:

python wgan_proto_sel.py --dataset APY --resSize 1024 --attSize 64 --nz 64 --nepoch 5 --bs 512 --lr 0.00005 --modeldir './APY_proto1024_model' --syn_num 2000 --nepoch_cls 40 --bs_cls 128 --lr_cls 0.0002

# Citation
If you find this project useful for your research, please cite this paper:

>@article{Liu2020ZeroShotLF,
  title={Zero-Shot Learning from Adversarial Feature Residual to Compact Visual Feature},
  author={Bo Liu and Qiulei Dong and Zhanyi Hu},
  journal={ArXiv},
  year={2020},
  volume={abs/2008.12962}
}

# Contact
If you have any questions, please contact liubo2017@ia.ac.cn

# Acknowledgement
This implementation refers to [this repo](https://github.com/akku1506/Feature-Generating-Networks-for-ZSL) and uses the datasets provided by [http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).

#
