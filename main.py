import pickle
import numpy as np
from deepModel.DeepNonLinearDynamicalSystem import *
import matplotlib.pyplot as plt


npzfile = np.load('./data/box.npz')
images = npzfile['images'].astype(np.float32)
images = (images > 0).astype('float32')
T = images.shape[1]
x_all_train = [images[i].reshape([T,-1]) for i in range(images.shape[0])]
u_dim = 2 ############
u_all_train = [np.zeros([T-1,u_dim])] * images.shape[0]
images=[]
npzfile = []
locations_all_train = [None]*len(x_all_train)
for i in range(len(x_all_train)):
    locations_all_train[i] = np.zeros([T,2], dtype=float)
    for j in range(T):
        [a,b] = np.where(x_all_train[i][j].reshape([32,32])[3:-2,3:-2] > 0)
        locations_all_train[i][j,:] = np.array([a[0]+2,b[0]], dtype=float)/32

'''
[x_all_train, u_all_train, states_train] = pickle.load(open('data/moving_particle_trajectory_train.data', 'rb'))
x_all_train = [ a.reshape([a.shape[0],-1]) for a in x_all_train ]
u_all_train = [ a[:-1].reshape([a.shape[0]-1,-1]) for a in u_all_train ]
'''

#####
'''
[x_all_validation, u_all_validation, states_validation] = pickle.load(open('data/moving_particle_trajectory_validation.data', 'rb'))
x_all_validation = ( a.reshpe([a.shape[0],-1]) for a in x_all_validation )
u_all_validation = ( a.reshpe([a.shape[0],-1]) for a in u_all_validation[:-1] )
'''
#####

deepNonLinearDynamicalSystem = DeepNonLinearDynamicalSystem()
#deepNonLinearDynamicalSystem.load_weights('./log/adam/3/4/')
#deepNonLinearDynamicalSystem.train(x_all_train, u_all_train, iter_EM_statr=4)
'''
T = 100
x_all_train = x_all_train[:T]
u_all_train = u_all_train[:T]
locations_all_train = locations_all_train[:T]
'''
deepNonLinearDynamicalSystem.train(x_all_train, u_all_train, locations_all_train)
#deepNonLinearDynamicalSystem.trainer.complementary_train(x_all_train, u_all_train)

'''
#---
dir_ = './log/box_complementary_run_3/12/0/'
deepNonLinearDynamicalSystem.load_weights(dir_)

tmp = pickle.load(open(dir_ + 'hist_observation_recons_loss.pkl','rb'))
hist_observation_recons_loss = np.zeros([len(tmp)])
for i in range(len(tmp)):
    hist_observation_recons_loss[i] = tmp[i][-1]

hist_EM_obj = pickle.load(open(dir_ + 'hist_EM_obj.pkl', 'rb'))

tmp = pickle.load(open(dir_ + 'hist_w_regularization_loss.pkl','rb'))
hist_w_regularization_loss = np.zeros([len(tmp)])
for i in range(len(tmp)):
    hist_w_regularization_loss[i] = tmp[i][-1]

[A,b,H,C,d,Q,R,mu_0,Sig_0] = pickle.load(open(dir_ + 'LDS_params.pkl','rb'))
#---
'''

'''
[x_all_test, u_all_test, states_test] = pickle.load(open('data/moving_particle_trajectory_test.data', 'rb'))
x_all_test = [ a.reshape([a.shape[0],-1]) for a in x_all_test ]
u_all_test = [ a[:-1].reshape([a.shape[0]-1,-1]) for a in u_all_test ]
'''


npzfile = np.load('./data/box_test.npz')
images = npzfile['images'].astype(np.float32)
images = (images > 0).astype('float32')
T = images.shape[1]
x_all_test = [images[i].reshape([T,-1]) for i in range(images.shape[0])]
u_dim = 2 ############
u_all_test = [np.zeros([T-1,u_dim])] * images.shape[0]
images=[]
npzfile = []


x_all_est = [None] * len(x_all_test)
w_all_est = [None] * len(x_all_test)
w_all_test = [None] * len(x_all_test)
for i in range(len(x_all_test)):
    print(i)
    x_all_est[i] = np.zeros(x_all_test[i].shape)
    x_all_est[i][:2] = x_all_test[i][:2]
    w_all_est[i] = np.zeros([x_all_test[i].shape[0], deepNonLinearDynamicalSystem.w_dim])
    w_all_est[i][:2] = deepNonLinearDynamicalSystem.observation_encoder.predict(x_all_est[i][:2])
    for j in range(2,x_all_test[i].shape[0]):
        [x_all_est[i][j], w_all_est[i][j]] = \
        deepNonLinearDynamicalSystem.predict(x_all_test[i][:j], u_all_test[i][:j])
    w_all_test[i] = deepNonLinearDynamicalSystem.observation_encoder.predict(x_all_test[i])


f, axarr = plt.subplots(2,1)
for i in range(0,1):
    for j in range(x_all_test[i].shape[0]):
        axarr[0].imshow(x_all_est[i][j].reshape([32,32]))
        axarr[1].imshow(x_all_test[i][j].reshape([32,32]))
        if j < 10:
            filename = '00' + str(j)
        elif j < 100:
            filename = '0' + str(j)
        else:
            filename = str(j)
        f.savefig('./out/' + str(i) + '/' + filename + '.jpg')
     

print(np.mean(np.linalg.norm(w_all_est[0] - w_all_test[0], axis=1)))
print(np.mean(np.linalg.norm(w_all_test[0], axis=1)))
print(np.mean(np.linalg.norm(w_all_test[0][1:] - w_all_test[0][:-1], axis=1)))


########################
'''
f, arr = plt.subplots(11,1)
delta = w_all_test[0][19] - w_all_test[0][10]
for i in range(11):
    arr[i].imshow(deepNonLinearDynamicalSystem.observation_decoder.predict((w_all_test[0][10] + i*delta/10).reshape([1,-1])).reshape([32,32]))
'''

########################