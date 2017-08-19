import numpy

from chainer import cuda
from chainer import optimizer


class STDProp(optimizer.GradientMethod):

    """Y. Ida et al. proposed optimization algorithm.
       SDProp (a.k.a. STDProp) without momentum.
       See also.  <http://arxiv.org/abs/1605.09593> or <https://www.ijcai.org/proceedings/2017/0267.pdf>
    """

    def __init__(self, alpha=0.001, gamma=0.99, eps=1e-5):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['m'] = param.grad
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        print("called STDPop")
        m = state['m']
        v = state['v']
        grad = param.grad

        v  = (self.gamma * v) + (self.gamma * (1.0 - self.gamma) * numpy.power(grad - m, 2))
        m  = (self.gamma * m) + ((1.0 - self.gamma) * grad)
        param.data = param.data - self.alpha * grad / (numpy.sqrt(v) + self.eps)

        state['m'] = m
        state['v'] = v

    def update_one_gpu(self, param, state):
        m = state['m']
        v = state['v']

        v, m = cuda.elementwise(
            'T grad, T alpha, T gamma, T eps, T m, T v',
            'T v_out, T m_out',
            '''v_out = (gamma * v) + (gamma * (1 - gamma) * (grad - m) * (grad - m));
               m_out = (gamma * m) + ((1 - gamma) * grad);
            ''',
            'stdprop1')(
               param.grad, self.alpha, self.gamma, self.eps, m, v)

        param.data = cuda.elementwise(
            'T grad, T alpha, T gamma, T eps, T v, T data',
            'T data_out',
            'data_out = data - alpha * grad / (sqrt(v) + eps)',
            'stdprop2')(
               param.grad, self.alpha, self.gamma, self.eps, v, param.data)

        state['m'] = m
        state['v'] = v

