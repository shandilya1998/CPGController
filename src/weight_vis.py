import matplotlib.pyplot as plt
import numpy as np
from train import *
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Visualize(object):
    def __init__(
        self,
        tsw, 
        tst,
        theta = 30, 
        exp_start = 10, 
        exp_num = 285,  
        dt = 0.001,
        nepochs = 1000,
        num_osc = 20,
        num_h = 50,
        num_out = 8,
        lr = 0.001,
        init = 'random',
    ):
        self.tsw = tsw
        self.tst = tst
        self.theta = theta
        self.exp = exp_start
        self.num = exp_num
        self.dt = dt
        self.nepochs = nepochs
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.lr = lr
        self.init = init

    def _create_train_obj(self, exp_id):
        return Train(
            self.dt, 
            5*(self.tst[exp_id]+self.tsw[exp_id]),
            self.nepochs,
            self.num_osc,
            self.num_h,
            self.num_out,
            exp_id,
            self.init,
            self.lr
        )

    def plot_weights(self, train, exp_id):
        plt.imsave('weights/exp{exp}/layer1_weights_real.png'.format(exp=exp_id), train.out_mlp.W1.real)
        plt.imsave('weights/exp{exp}/layer2_weights_real.png'.format(exp=exp_id), train.out_mlp.W2.real)
        plt.imsave('weights/exp{exp}/layer1_weights_imag.png'.format(exp=exp_id), train.out_mlp.W1.imag)
        plt.imsave('weights/exp{exp}/layer2_weights_imag.png'.format(exp=exp_id), train.out_mlp.W2.imag)    

    def visualize_params(self, vis_exp_id):
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(tst/(tst+tsw))
        axes[0].set_ylabel('duty factor')
        axes[0].set_xlabel('training instance')
        axes[0].set_title('duty factor trend in visualization experiment')
        axes[1].plot(2*6*self.theta*2*np.pi/(360*(tst+tsw)))
        axes[1].set_ylabel('calculated speed')
        axes[1].set_xlabel('training instance')
        axes[1].set_title('calculated speed trend in visualization experiment') 
        fig.set_figheight(15)
        fig.set_figwidth(7.5)
        fig.savefig('weight_visualization_exp{exp}_parameter_trends.png'.format(exp=vis_exp_id) )

    def __call__(self, vis_exp_id):
        self.visualize_params(vis_exp_id)
        i = self.exp
        while(i-self.exp<self.num):
            path = os.path.join('weights','exp{exp}'.format(exp=i))
            if(not os.path.exists(path)):
                os.mkdir(path)
                train = self._create_train_obj(i)
                train(self.tst[i-self.exp], self.tsw[i-self.exp], self.theta)
                self.plot_weights(train, i)
            i+=1
            plt.close('all')

tst = np.arange(60, 400, 10) 
tsw = np.arange(20, 360, 10)
#vis = Visualize(tsw, tst, exp_num = 36)
#vis(1)

n_components = 0.8
lr = 150
prp = 30
vis_exp=1
weights = ['w1_out_mlp.npy', 'w2_out_mlp.npy']
def get_tsne():
    X = [np.zeros((33, 1000)), np.zeros((33, 1000)), np.zeros((33, 400)), np.zeros((33, 400))]
    X_pca = []
    Y = []
    for i in range(10, 34):
        path = os.path.join('weights', 'exp{exp}'.format(exp=i))
        path1 = os.path.join(path, weights[0])
        path2 = os.path.join(path, weights[1])  
        W1 = np.load(path1)
        W2 = np.load(path2)
        X[0][i-10]=W1.real.flatten()
        X[1][i-10]=W1.imag.flatten()
        X[2][i-10]=W2.real.flatten()
        X[3][i-10]=W2.imag.flatten()
    pca = [PCA(n_components=n_components).fit_transform(X[i]) for i in range(4)]
    for i in range(4):
        x = []
        for j in range(33):
            #print(pca[i].shape)
            x.append(pca[i][j])
        x = np.array(x)
        X_pca.append(x)
    fig, axes = plt.subplots(4, 1)
    tsne = TSNE(
        n_components=2,
        learning_rate = lr,
        perplexity = prp
                ).fit_transform(X_pca[0])
    #print(tsne.shape)
    axes[0].scatter(tsne[:, 0], tsne[:, 1])
    axes[0].set_title('layer 1 real part')
    tsne = TSNE(
        n_components=2,
        learning_rate = lr,
        perplexity = prp
                ).fit_transform(X_pca[1])
    #print(tsne.shape)
    axes[1].scatter(tsne[:, 0], tsne[:, 1])
    axes[1].set_title('layer 1 imaginary part')
    tsne = TSNE(
        n_components=2,
        learning_rate = lr,
        perplexity = prp
                ).fit_transform(X_pca[2])
    #print(tsne.shape)
    axes[2].scatter(tsne[:, 0], tsne[:, 1])
    axes[2].set_title('layer 2 real part')
    tsne = TSNE(
        n_components=2,
        learning_rate = lr,
        perplexity = prp
                ).fit_transform(X_pca[3])
    #print(tsne.shape)
    axes[3].scatter(tsne[:, 0], tsne[:, 1])
    axes[3].set_title('layer 2 imaginary part')
    fig.set_figheight(30)
    fig.set_figwidth(7.5)
    plt.show()
    fig.savefig('tsne_weights_vis_exp{exp}.png'.format(exp=vis_exp)) 

get_tsne()
