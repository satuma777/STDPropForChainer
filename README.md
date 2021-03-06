# SDPropForChainer
This is a [chainer](http://chainer.org/) implementation of the paper [Adaptive Learning Rate via Covariance Matrix Based Preconditioning for Deep Neural Networks](https://www.ijcai.org/proceedings/2017/0267.pdf)(Y. Ida et al., International Joint Conference on Artificial Intelligence(IJCAI), 2017).
This paper presents SDProp: a novel training algorithm for deep neural networks.

Since [the previous version of this paper](https://arxiv.org/abs/1605.09593) refers to SDProp as STDProp, the name of STDProp is used in the code.

## Requirements
Minimum requirements:

- Python 2.7 or 3.4 later (This has 2 and 3 compatibility)
- Chainer(<= 1.24.0) and minimum dependencies

## Usage
Basic usage:
```
import stdprop
```

SDProp with momentum:
```
import momentum_stdprop
```

They have been created by extending [GradientMethod](http://docs.chainer.org/en/stable/_modules/chainer/optimizer.html#GradientMethod) class.

