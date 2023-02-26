"""
Author : Neeraja Abhyankar
Created : October 2019
"""

from __future__ import print_function
from __future__ import division

import os
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import seaborn as sns
sns.set()

from sklearn.neural_network import BernoulliRBM
from sklearn import datasets

#%% Creation : method 1

# an 8-bit color image has Red as 3 bits, Green as 3 bits, and Blue as 2.
# imshow accepts arrays of the form MxNx3, where the last dimension has RGB values between 0-255.
# however, we restrict colors to take on a limited set of values between 0-255
R = np.random.randint(8, size=(32,32))*32
G = np.random.randint(8, size=(32,32))*32
B = np.random.randint(4, size=(32,32))*64
I = np.dstack((R,G,B))
#note that there are 8*8*4 = 256 distinct RGB values that each pixel can have.

plt.figure(); plt.axis("off")
plt.imshow(I)

#%% Creation : method 2

M = np.random.randint(256, size=(32,32))

plt.figure(); plt.axis("off")
plt.imshow(np.dstack(((np.floor(M/32)%8)*32, (np.floor(M/4)%8)*32, (M%4)*64)).astype(int))

plt.figure(); plt.axis("off")
plt.imshow(M, cmap='binary') # produces grayscale

#%% Solution : method 1 -- choosing from a number of random images

N = 100
allimages = np.empty((N, 32, 32, 3), dtype=int)
for n in range(N):
    M = np.random.randint(256, size=(32,32))
    allimages[n] = np.dstack(((np.floor(M/32)%8)*32, (np.floor(M/4)%8)*32, (M%4)*64))

# random image generation
for g in range(10):
    
    # the source of true randomness
    M = allimages[np.random.randint(N)]
    
    plt.figure(); plt.axis("off")
    plt.imshow(M)

# save plots
plt.figure(figsize=(25, 5))
for g in range(5):
    plt.subplot(1, 5, g+1)
    
    M = allimages[np.random.randint(N)]
    plt.axis("off")
    plt.imshow(M)

plt.savefig('hw3p8-sol1.pdf', dpi=500)

#%% Solution : method 2 -- using a pseudo-random number generator

m = 1051
c = 0
a = 7

# random image generation
for g in range(10):
    
    # the source of true randomness
    seed = np.random.randint(m-1)

    M = np.empty(1024, dtype=int)
    x=seed
    for i in range(1024):
        x = (x*a + c)%m
        M[i] = x
    
    M = M.reshape((32,32))
    plt.figure(); plt.axis("off")
    plt.imshow(np.dstack(((np.floor(M/32)%8)*32, (np.floor(M/4)%8)*32, (M%4)*64)).astype(int))

# save plots
plt.figure(figsize=(25, 5))
for g in range(5):
    plt.subplot(1, 5, g+1)

    seed = np.random.randint(m-1)
    M = np.empty(1024, dtype=int)
    x=seed
    for i in range(1024):
        x = (x*a + c)%m
        M[i] = x
    
    M = M.reshape((32,32))
    plt.axis("off")
    plt.imshow(np.dstack(((np.floor(M/32)%8)*32, (np.floor(M/4)%8)*32, (M%4)*64)).astype(int))

plt.savefig('hw3p8-sol2.pdf', dpi=500)

#%% Solution : method 3 -- using graphical models

#%% Gibb's sampler for a B&W image with neighboring connections

L = 32

def neighbors4(i0,j0):
    """ Returns a numpy list of pairs [i,j] which are defined to be connected to [i0,j0]
        by edges of a graph
    """
    nb = np.empty((0,2), dtype=int)
    if (i0 > 0): nb = np.vstack((nb, [i0-1, j0]))
    if (j0 > 0): nb = np.vstack((nb, [i0, j0-1]))
    if (i0 < L-1): nb = np.vstack((nb, [i0+1, j0]))
    if (j0 < L-1): nb = np.vstack((nb, [i0, j0+1]))
    #return np.array([[i0-1, j0](i0 > 0), [i0, j0-1](j0 > 0), [i0+1, j0](i0 < L-1), [i0, j0+1](j0 < L-1)])
    #np.array([[max(i0-1,0), j0], [i0, max(j0-1, 0)], [min(i0+1,L-1), j0], [i0, min(j0+1,L-1)]])
    return nb

def neighbors8(i0,j0):
    """ Returns a numpy list of pairs [i,j] which are defined to be connected to [i0,j0]
        by edges of a graph
    """
    nb = np.empty((0,2), dtype=int)
    if (i0 > 0): nb = np.vstack((nb, [i0-1, j0]))
    if (j0 > 0): nb = np.vstack((nb, [i0, j0-1]))
    if (i0 < L-1): nb = np.vstack((nb, [i0+1, j0]))
    if (j0 < L-1): nb = np.vstack((nb, [i0, j0+1]))
    if (i0 > 0) and (j0 > 0): nb = np.vstack((nb, [i0-1, j0-1]))
    if (i0 > 0) and (j0 < L-1): nb = np.vstack((nb, [i0-1, j0+1]))
    if (i0 < L-1) and (j0 > 0): nb = np.vstack((nb, [i0+1, j0-1]))
    if (i0 < L-1) and (j0 < L-1): nb = np.vstack((nb, [i0+1, j0+1]))
    return nb

#%% A case where things should ideally converge

beta = 0.2
M_init = np.random.choice([-1,1], (L,L))
#M_init = np.ones((32, 32))

plt.figure(); plt.axis("off")
plt.imshow(M_init, cmap='binary') # produces grayscale
M = M_init.copy()
M_prev = np.zeros((L,L))

niter = 0
#while abs(np.sum(M - M_prev)) > 8:
while niter < 1000:
    M_prev = M.copy()
    niter += 1
    for i in range(L):
        for j in range(L):
            wij = np.sum(M[neighbors4(i,j)]*M[i,j])
            #wij = np.sum(M[neighbors(i,j)])
            pij = math.exp(beta*wij)/(math.exp(beta*wij) + math.exp(-beta*wij))
            M[i,j] = int(np.random.random() > pij)*2 - 1

#    plt.figure(); plt.axis("off")
#    plt.imshow(M, cmap='binary')

plt.figure(); plt.axis("off")
plt.imshow(M*2-1, cmap='binary') # produces grayscale

#%% For MNIST : courtesy Stackoverflow

""" A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a
    probability distribution over its set of inputs.
    This BernoulliRBM has *n_components* binary hidden units and *n_features* binary visible units.
    .gibbs() accepts a matrix of shape (n_samples, n_features)
"""

def show_digits(im, title):
    """ im.shape = (n_samples, 64)
    """
    plt.figure(figsize=(5,5))
    plt.gray()
    for i in range(im.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(np.reshape(im[i,:], (8,8)))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def rbm():
    """
    """
    
    ## preproc images
    digits = datasets.load_digits()
    bindigit_trn = np.asarray(digits.data, 'float32') # shape (1797, 64) = (n_samples, n_features)
    for i in range(len(bindigit_trn)):
        bindigit_trn[i,:] = bindigit_trn[i,:] / np.max(bindigit_trn[i,:]) # normalize every sample s.t. pixels in [0,1]
    print(bindigit_trn.shape) # shape (1797, 64)
    
    ## take the first 100
    digits = bindigit_trn[:100,:]
    print(digits.shape) # shape (100, 64) => 100 images
    show_digits(digits, 'original digits')
    
    ## find the underlying probability distribution from which they're sampled
    rbm = BernoulliRBM(n_iter= 10, learning_rate = 0.1, n_components = 10, random_state=0, verbose=True)
    rbm.fit(bindigit_trn)
    print(rbm.components_.shape) # shape (10, 64)
    
    ## sample from this distribution
    """ this is based on the original digits that we had!
        we take every sample, and take a single step towards the steady state distribution...
        we haven't gotten rid of the dependency of starting from a random (high-entropy) starting point here!
    """
    digits_new = digits.copy() # rbm.components_.copy()
    for j in range(10000):
        for i in range(100):
            digits_new[i,:] = rbm.gibbs(digits_new[i,:])
    print(digits_new.shape) # shape (100, 64)
    show_digits(digits_new, 'sampled digits')
    
    ## find the weights of the network
    weights = rbm.components_
    return weights

weights = rbm()
show_digits(weights, 'weights')


## a hack
#perm = np.random.permutation(64)
#show_digits(np.array([pix[perm] for pix in digits_new]), 'sampled digits')


#%% For CIFAR10

def unpickle(file):
    """ Returns a dict containing
        
        data : a 10000x3072 numpy array of uint8s.
            Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
        
        labels : a list of 10000 numbers in the range 0-9.
            The number at index i indicates the label of the ith image in the array data.
            ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """
    with open(file, 'rb') as fo:
        cifardict = pickle.load(fo, encoding='bytes')
    return cifardict


def show_im(im):
    """ im.shape = (n_samples, 1024)
    """
    perm = np.random.permutation(1024)
    plt.figure(figsize=(5,5))
    plt.gray()
    for i in range(im.shape[0]):
        plt.subplot(10,10,i+1)
        plt.imshow(np.reshape(im[i,:][perm], (32,32)))
        plt.axis('off')
    plt.show()


def plotimages(Mset):
    """ Mset.shape = (Nsamples, 1024) with values = integers in [0,255]
        Plots the first 5 images from this
    """

    plt.figure(figsize=(25, 5))
    for i in range(5):
        M = Mset[i].reshape(32,32)
        
        plt.subplot(1, 5, i+1); plt.axis("off")
        plt.imshow(np.dstack(((np.floor(M/32)%8)*32, (np.floor(M/4)%8)*32, (M%4)*64)).astype(int))

## preproc images
dir_path = os.getcwd() + '/infotheory-aut19-TA/'
cifar = unpickle(dir_path + 'cifar-10-batches/' + 'data_batch_1')
cdata = cifar[b'data']
clabels = cifar[b'labels']


trainbinred = np.floor(cdata[:,:1024]/128).astype(int)

## find the underlying probability distribution from which they're sampled
rbm = BernoulliRBM(n_iter= 10, learning_rate = 0.1, n_components = 20, random_state=0, verbose=True)
rbm.fit(trainbinred)
print(rbm.components_.shape)

## sample from this distribution
""" this is based on the original digits that we had!
    we take every sample, and take a single step towards the steady state distribution...
    we haven't gotten rid of the dependency of starting from a random (high-entropy) starting point here!
"""
gibbsop = trainbinred[:100].copy() # rbm.components_.copy()
for j in range(10):
    for i in range(100):
        gibbsop[i,:] = rbm.gibbs(gibbsop[i,:])
print(gibbsop.shape) # shape (100, 1024)

show_im(gibbsop)


## the stuff  below outputs non-binary values
#randop = np.empty((100,1024), dtype=int)
#for i in range(5):
#    seedflat = np.random.randint(2, size=(20))
#    randop[i,:] = np.matmul(seedflat, weights)
#show_im(randop)

"""
cifar10 is nsamples x 3072 \in [0,255]
rbm wants a binary grid : nsamples x 1024*256
ultimately turn it  into a nsamples x 1024 \in [0,255] or \in {0,1} worst case

train_binary = np.empty((10000, 1024*8), dtype=int)
for b in range(8):
    train_binary[:,1024*b:1024*(b+1)] = (cdata//(128/(2**b))%2).astype(int)

plt.imshow(np.dstack(( (M[:,:,5]+M[:,:,6]+4*M[:,:,7])*32, (M[:,:,2]+M[:,:,3]+4*M[:,:,4])*32, (M[:,:,0]+2*M[:,:,1])*64 )).astype(int))

"""



def plot_natural(Mset):
    """ Mset.shape = (Nsamples, 1024) with binary values
        Plots the first 5 images from this
    """
    plt.figure(figsize=(25, 5))
    for i in range(5):
        M = Mset[i].reshape(32,32)
        plt.subplot(1, 5, i+1); plt.axis("off")
        plt.imshow(M)

def plot_permuted(Mset):
    """ Mset.shape = (Nsamples, 1024) with binary values
        Plots the first 5 images from this
    """
    perm = np.random.permutation(1024)
    plt.figure(figsize=(25, 5))
    for i in range(5):
        M = Mset[i][perm].reshape(32,32)
        plt.subplot(1, 5, i+1); plt.axis("off")
        plt.imshow(M)

## load
cdata = pickle.load(open(os.getcwd()+'/infotheory-aut19-TA/cifar-10-batches/data_batch_1', 'rb'), encoding='bytes')[b'data'][:10000,:1024]

## convert to binary
train_binary = (cdata//128).astype(int)
plot_natural(train_binary)

## learn weights to hidden nodes
rbm = BernoulliRBM(n_iter = 20, learning_rate = 0.1, n_components = 25, random_state=0, verbose=True)
rbm.fit(train_binary)

## sample from this distribution
gibbsop = trainbinred[5:10].copy()
for j in range(1000):
    for i in range(5):
        gibbsop[i,:] = rbm.gibbs(gibbsop[i,:])

plot_natural(gibbsop)

plot_permuted(gibbsop)

#%% CIFAR examples

import pandas as  pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1

pd_tr = pd.DataFrame()
tr_y = pd.DataFrame()

data = unpickle(os.getcwd()+'/infotheory-aut19-TA/cifar-10-batches/data_batch_1')
pd_tr = pd_tr.append(pd.DataFrame(data[b'data']))
tr_y = tr_y.append(pd.DataFrame(data[b'labels']))
pd_tr['labels'] = tr_y

tr_x = np.asarray(pd_tr.iloc[:, :3072])
tr_y = np.asarray(pd_tr['labels'])
ts_x = np.asarray(data[b'data'])
ts_y = np.asarray(data[b'labels'])

def plot_CIFAR(ind):
    arr = tr_x[ind]
    R = arr[0:1024].reshape(32,32)#/255.0
    G = arr[1024:2048].reshape(32,32)#/255.0
    B = arr[2048:].reshape(32,32)#/255.0

    img = np.dstack((R,G,B))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(img,interpolation='bicubic')

for ii in range(10):
    plot_CIFAR(ii)

#%% For random images

def plotimages(Mset):
    """ Mset.shape = (Nsamples, 1024) with values = integers in [0,255]
        Plots the first 5 images from this
    """

    plt.figure(figsize=(25, 5))
    for i in range(5):
        M = Mset[i].reshape(32,32)
        
        plt.subplot(1, 5, i+1); plt.axis("off")
        plt.imshow(np.dstack(((np.floor(M/32)%8)*32, (np.floor(M/4)%8)*32, (M%4)*64)).astype(int))
    
    #plt.savefig('hw3p8-sol3.pdf', dpi=500)


Nseeds = 20
train = np.random.randint(256, size=(Nseeds, 1024))
plotimages(train)

## find the underlying probability distribution from which they're sampled
rbm = BernoulliRBM(n_iter= 25, learning_rate = 0.1, n_components = 10, random_state=0, verbose=True)
rbm.fit(train)
weights = rbm.components_
print(weights.shape) # shape (20, 1024)

randset = np.empty((5,1024), dtype=int)
for i in range(5):
    seedflat = np.random.randint(256, size=(10))
    randset[i,:] = np.matmul(seedflat, weights)
plotimages(randset)


#%% The 256 spectrum

seq = np.arange(1024); seq = seq//4;
nat = seq.reshape(32,32)
plt.figure(); plt.axis("off")
plt.imshow(np.dstack(((np.floor(nat/32)%8)*32, (np.floor(nat/4)%8)*32, (nat%4)*64)).astype(int))

perm = np.random.permutation(1024)
randnat = seq[perm].reshape(32,32)
plt.figure(); plt.axis("off")
plt.imshow(np.dstack(((np.floor(randnat/32)%8)*32, (np.floor(randnat/4)%8)*32, (randnat%4)*64)).astype(int))

#%%

"""
##TODO

- Just go through GM slides
- Bether approximation + https://www.cs.cmu.edu/~epxing/Class/10708-05/Slides/lecture17.pdf
    should be good resources
    
- See whether sklearn's Gibb's has any notion of neighborhood
- If not, default to the model we wrote ..
    from https://www2.isye.gatech.edu/~brani/isyebayes/bank/handout16.pdf
    or http://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-7/07-gibbs.pdf
- Check whether the neighboring thing is sufficient or whether I actually have to encode edge potentials etc.
- My guess is : you don't. That's why they have that exp formula thing
- See what you get with 8 neighbors .. wiat till you get anything that looks like a natural image. Apply a perm to it!
- Figure out how to sample from it and get something different every time!!!!

- Some promising implementations :
    https://rajeshrinet.github.io/blog/2014/ising-model/
    https://compphys.go.ro/metropolis-for-2d-ising-and-spin-block-renormalization/ (this looks like its sampling!!!)
    
"""
