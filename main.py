import pickle
import numpy as np
from deepModel.DeepNonLinearDynamicalSystem import *
import matplotlib.pyplot as plt

[x_all_train, u_all_train, states_train] = pickle.load(open('data/moving_particle_trajectory_train.data', 'rb'))
x_all_train = [ a.reshape([a.shape[0],-1]) for a in x_all_train ]
u_all_train = [ a[:-1].reshape([a.shape[0]-1,-1]) for a in u_all_train ]

#####
'''
[x_all_validation, u_all_validation, states_validation] = pickle.load(open('data/moving_particle_trajectory_validation.data', 'rb'))
x_all_validation = ( a.reshpe([a.shape[0],-1]) for a in x_all_validation )
u_all_validation = ( a.reshpe([a.shape[0],-1]) for a in u_all_validation[:-1] )
'''
#####

deepNonLinearDynamicalSystem = DeepNonLinearDynamicalSystem()
deepNonLinearDynamicalSystem.train(x_all_train, u_all_train)

#---
dir_ = './log/adam/9/4/'
deepNonLinearDynamicalSystem.load_weights(dir_)

hist_observation_recons_loss = pickle.load(open(dir_ + 'hist_observation_recons_loss.pkl','rb'))
hist_observation_recons_loss = np.concatenate(hist_observation_recons_loss)

hist_EM_obj = pickle.load(open(dir_ + 'hist_EM_obj.pkl', 'rb'))

hist_w_regularization_loss = pickle.load(open(dir_ + 'hist_w_regularization_loss.pkl','rb'))
hist_w_regularization_loss = np.concatenate(hist_w_regularization_loss)

[A,b,H,C,d,Q,R,mu_0,Sig_0] = pickle.load(open(dir_ + 'LDS_params.pkl','rb'))
#---



[x_all_test, u_all_test, states_test] = pickle.load(open('data/moving_particle_trajectory_test.data', 'rb'))
x_all_test = [ a.reshape([a.shape[0],-1]) for a in x_all_test ]
u_all_test = [ a[:-1].reshape([a.shape[0]-1,-1]) for a in u_all_test ]

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


f, axarr = plt.subplots(2, sharex=True)
axarr[0].imshow(x_all_est[0][100].reshape([40,40]))
axarr[1].imshow(x_all_test[0][100].reshape([40,40]))