import pickle
import numpy as np
from deepModel.DeepNonLinearDynamicalSystem import *

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



[x_all_test, u_all_test, states_test] = pickle.load(open('data/moving_particle_trajectory_test.data', 'rb'))
x_all_test = [ a.reshape([a.shape[0],-1]) for a in x_all_test ]
u_all_test = [ a[:-1].reshape([a.shape[0]-1,-1]) for a in u_all_test ]

x_all_est = [None] * len(x_all_test)
for i in range(len(x_all_test)):
    print(i)
    x_all_est[i] = np.zeros(x_all_test[i].shape)
    x_all_est[i][:1] = x_all_test[i][:1]
    for j in range(2,x_all_test[i].shape[0]):
        x_all_est[i][j] = deepNonLinearDynamicalSystem.predict(x_all_test[i][:j-1], u_all_test[i][:j-1])
