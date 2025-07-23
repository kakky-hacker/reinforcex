from numpy.linalg import norm
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import training
from chainerrl.misc.batch_states import batch_states
from chainer import serializers
import numpy as np
import matplotlib.pyplot as plt
from random import random
import os

def _add_baseline(x, b):
    return ((x / b) ** 0.4 - 0.05) / 0.95

class _Model(chainer.Chain):
    def __init__(self, input_size, output_size, n_hidden_channels):
        w = chainer.initializers.HeNormal(scale=1.0)
        super(_Model, self).__init__()        
        with self.init_scope():
            self.l1 = L.Linear(input_size, n_hidden_channels, initialW = w)
            self.l2 = L.Linear(n_hidden_channels, n_hidden_channels, initialW = w)
            self.l3 = L.Linear(n_hidden_channels, n_hidden_channels, initialW = w)
            self.l4 = L.Linear(n_hidden_channels, output_size, initialW = w)

    def __call__(self, x, y):
        return F.mean_squared_error(self.predict(x), y)

    def predict(self, s):
        h = F.relu(self.l1(s))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return self.l4(h)

class RND:
    def __init__(self, obs_size, batchsize=128, alpha=5e-3, epoch=4, gpu=None, reward_coef=1.0, padding=0.0):
        self.obs_size = obs_size
        self.rfn = _Model(obs_size, 3, obs_size*20)
        self.pn = _Model(obs_size, 3, obs_size*20)
        self.gpu = gpu
        if gpu != None:
            cuda.get_device(gpu).use()
            self.rfn.to_gpu()
            self.pn.to_gpu()
        self.optimizer = chainer.optimizers.Adam(alpha=alpha, eps=0.1)
        self.optimizer.setup(self.pn)
        self.batchsize = batchsize
        self.epoch = epoch
        self.phi = lambda x: x.astype(np.float32, copy=False)
        self.batch_states = batch_states
        self.xp = self.rfn.xp
        self.baseline = self._calc_baseline_internal_reward()
        self.reward_coef = reward_coef
        self.padding = padding
        self.stat = []

    def _calc_baseline_internal_reward(self):
        state_set = np.array(2*np.random.rand(100, self.obs_size) - 1, dtype=np.float32)
        batch_xs = self.batch_states(state_set, self.xp, self.phi)
        rf = cuda.to_cpu(self.rfn.predict(batch_xs).data)
        pf = cuda.to_cpu(self.pn.predict(batch_xs).data)
        return np.average([norm(_pf - _rf) for _rf, _pf in zip(rf, pf)]) + 1e-9
        
    def update(self, state_set):
        state_set = np.array(state_set).astype(np.float32)
        batch_xs = self.batch_states(state_set, self.xp, self.phi)
        rf = np.array(cuda.to_cpu(self.rfn.predict(batch_xs).data)).astype(np.float32) 
        train = chainer.datasets.TupleDataset(state_set, rf)
        train_iter = chainer.iterators.SerialIterator(train, self.batchsize)
        updater = training.StandardUpdater(train_iter, self.optimizer, device=self.gpu)
        trainer = training.Trainer(updater, (self.epoch, 'epoch'))
        trainer.run()
        self.baseline = self._calc_baseline_internal_reward()

    def calc_internal_reward(self, state_set, baseline=True):
        batch_xs = self.batch_states(state_set, self.xp, self.phi)
        rf = cuda.to_cpu(self.rfn.predict(batch_xs).data)
        pf = cuda.to_cpu(self.pn.predict(batch_xs).data)
        if baseline:
            reward = [max(0, min(1.0, _add_baseline(norm(_pf - _rf), self.baseline))) * self.reward_coef for _rf, _pf in zip(rf, pf)]
        else:
            reward = [max(0, min(1.0, norm(_pf - _rf)))  * self.reward_coef for _rf, _pf in zip(rf, pf)]
        self.stat.append(np.average(reward))
        return reward

    def get_statistics(self, reset=True):
        if len(self.stat) > 0:
            average_internal_reward = np.average(self.stat)
            if reset:
                self.stat = []
        else:
            average_internal_reward = None
        data = {"average_internal_reward":average_internal_reward}
        return data

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.pn.to_cpu()
        self.rfn.to_cpu()
        serializers.save_npz(path + "/pn.npz", self.pn)
        serializers.save_npz(path + "/rfn.npz", self.rfn)
        if self.gpu != None:
            cuda.get_device(self.gpu).use()
            self.rfn.to_gpu()
            self.pn.to_gpu()

    def load(self, path):
        self.pn.to_cpu()
        self.rfn.to_cpu()
        serializers.load_npz(path + "/pn.npz", self.pn)
        serializers.load_npz(path + "/rfn.npz", self.rfn)
        if self.gpu != None:
            cuda.get_device(self.gpu).use()
            self.rfn.to_gpu()
            self.pn.to_gpu() 
        self.baseline = self._calc_baseline_internal_reward()
        
def main():
    n = 3000
    window = 10
    rnd = RND(13, batchsize=32, epoch=3, alpha=1e-3, reward_coef=1.0, gpu=0, padding=0.0)
    _y1, _y2 = [], []
    y1, y2 = [], []
    b = []
    state_set = np.array(2*np.random.rand(100, 13) - 1, dtype=np.float32)
    for i in range(n):
        #state_set = np.array(2*np.random.rand(10, 13) - 1, dtype=np.float32)
        _y1.append(np.average(rnd.calc_internal_reward(state_set, baseline=True)))
        _y2.append(np.average(rnd.calc_internal_reward(state_set, baseline=False)))
        b.append(rnd.baseline)
        rnd.update(state_set)
        if i % window == 0:
            y1.append(np.average(_y1))
            y2.append(np.average(_y2))
            _y1, _y2 = [], []
        if i % 200 == 0:
            state_set += np.array((2*np.random.rand(100, 13) - 1) * 0.05, dtype=np.float32)
        if i % 1000 == 0:
            state_set = np.array(2*np.random.rand(100, 13) - 1, dtype=np.float32)
    x = [i*window for i in range(len(y1))]
    plt.plot(x, y1, label="with baseline")
    plt.plot(x, y2, label="without baseline")
    #plt.plot([i for i in range(n)], b, label="baseline")
    plt.legend()
    plt.show()

main()
