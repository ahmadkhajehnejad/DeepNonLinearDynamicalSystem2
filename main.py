import pickle
from deepModel.DeepNonLinearDynamicalSystem import *

[x_all_train, u_all_train, states_all] = pickle.load(open('data/moving_particle_trajectory_train.data', 'rb'))
x_all_train = [ a.reshape([a.shape[0],-1]) for a in x_all_train ]
u_all_train = [ a[:-1].reshape([a.shape[0]-1,-1]) for a in u_all_train ]

'''
[x_all_validation, u_all_validation, states_validation] = pickle.load(open('data/moving_particle_trajectory_validation.data', 'rb'))
x_all_validation = ( a.reshpe([a.shape[0],-1]) for a in x_all_validation )
u_all_validation = ( a.reshpe([a.shape[0],-1]) for a in u_all_validation[:-1] )
'''

deepNonLinearDynamicalSystem = DeepNonLinearDynamicalSystem()
deepNonLinearDynamicalSystem.train(x_all_train, u_all_train)