# https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/38125

from keras.optimizers import Optimizer
import keras.backend as K

class SGDAccumulate(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                nesterov=False, accum_iters=5, **kwargs):
        super(SGDAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0., name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.accum_iters = K.variable(accum_iters, name='accum_iters')
        self.initial_decay = decay
        self.nesterov = nesterov

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        gradients = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + moments

        for p, g, m, gg in zip(params, grads, moments, gradients):

            flag = K.equal(self.iterations % self.accum_iters, 0)

            if K.eval(flag):
                flag = 1

            gg_t = (1 - flag) * (gg + g)

            v = self.momentum * m - lr * \
                K.square((gg + flag * g) / self.accum_iters)  # velocity

            self.updates.append(K.update(m, flag * v + (1 - flag) * m))
            self.updates.append((gg, gg_t))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'accum_iters': float(K.get_value(self.accum_iters)),
                  'nesterov': self.nesterov}
        base_config = super(SGDAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))