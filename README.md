# SDPropForChainer
This is a [chainer](http://chainer.org/) implementation of the paper [Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks](https://www.ijcai.org/proceedings/2017/0267.pdf)(Y. Ida et al., International Joint Conference on Artificial Intelligence(IJCAI), 2017).
This paper presents SDProp: a novel training algorithm for deep neural networks.

Since [the previous version of this paper](https://arxiv.org/abs/1605.09593) refers to SDProp as STDProp, the name of STDProp is used in the code.

## Usage
Basic usage:
```
import stdprop
```

SDProp wiht momentum:
```
import momentum_stdprop
```

They have been created by extending [GradientMethod](http://docs.chainer.org/en/stable/_modules/chainer/optimizer.html#GradientMethod) class.

## Citation
If you find this code useful for your research, please cite:
```
@inproceedings{ijcai2017-267,
  author    = {Yasutoshi Ida, Yasuhiro Fujiwara, Sotetsu Iwamura},
  title     = {Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks},
  booktitle = {Proceedings of the Twenty-Sixth International Joint Conference on
               Artificial Intelligence, {IJCAI-17}},
  pages     = {1923--1929},
  year      = {2017},
  doi       = {10.24963/ijcai.2017/267},
  url       = {https://doi.org/10.24963/ijcai.2017/267},
}
```
