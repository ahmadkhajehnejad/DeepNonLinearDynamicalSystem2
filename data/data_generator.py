from moving_particle_mdp import MovingParticleMDP
import pickle
import os
import numpy as np
import time

'''
def get_plane_dataset_trajectory(filename,
                      num_epochs=1,
                      num_samples=100,
                      W=40, H=40,
                      overwrite_datasets=True,
                      **kwargs):
    """
    Create a dataset for the Plane with obstacles MDP.
    """
    if not os.path.exists(filename) or overwrite_datasets:
        print('Creating plane dataset', filename)
        mdp = plane_obstacles_mdp.PlaneObstaclesMDP(H=H, W=W)

        X = [None] * num_epochs
        U = [None] * num_epochs
        True_state = [None] * num_epochs
        for epoch in range(num_epochs):
            print(epoch)
            X[epoch] = np.zeros((num_samples, 1, H, W), dtype='float32')
            U[epoch] = np.zeros((num_samples, mdp.action_dim), dtype='float32')
            True_state[epoch] =  np.zeros((num_samples,2), dtype='float32')
            #state = np.array([6,6])
            #s = (state , mdp.render(state))
            s = mdp.sample_random_state()
            #y_start = mdp.render(s[0], 'rectangle')
            for i in range(num_samples):
                valid_action = False
                while not valid_action:
                    a = np.array([np.random.randint(mdp.arange_min[0],
                                                    mdp.arange_max[0]+1),
                                  np.random.randint(mdp.arange_min[1],
                                                    mdp.arange_max[1]+1)])
                    a = np.float64(a)
                    a_noisy = a + np.array([np.random.randint(mdp.anoiserange_min[0],
                                                    mdp.anoiserange_max[0]+1),
                                  np.random.randint(mdp.anoiserange_min[1],
                                                    mdp.anoiserange_max[1]+1)])
                    valid_action = mdp.is_valid_action(s, a) and mdp.is_valid_action(s, a_noisy)
    ##                action_noise = np.random.normal(loc=0,scale=1,size = a.shape)
    ##                aa = a + action_noise
    ##                valid_action = mdp.is_valid_action(s, aa)
                ###a = np.array([2.2,2.1])
                ns = mdp.transition_function(s, a_noisy)
                
                a /= np.max(mdp.arange_max)
    
                X[epoch][i, :] = s[1]
                U[epoch][i, :] = a
                True_state[epoch][i, :] = s[0]
                s = ns
                    
        pickle.dump((X, U, True_state), open(filename, 'wb'))
        return X, U
    else:
        print('Loading plane dataset', filename)
        return pickle.load(open(filename, 'rb'))
        

get_plane_dataset_trajectory('plane_random_trajectory_train', num_samples = 15, num_epochs = 400)
get_plane_dataset_trajectory('plane_random_trajectory_test', num_samples = 50, num_epochs = 20)

mdp = plane_obstacles_mdp.PlaneObstaclesMDP(H=40, W=40)
[s,x] = mdp.get_all_states()
pickle.dump([s,x], open('all_states', 'wb'))
'''

def get_moving_particle_dataset_trajectory(filename,
                      num_epochs=1,
                      num_samples=100,
                      W=40, H=40,
                      overwrite_datasets=True,
                      **kwargs):
    if not os.path.exists(filename) or overwrite_datasets:
        print('Creating plane dataset', filename)
        mdp = MovingParticleMDP(H=H, W=W)

        S = [None] * num_epochs
        X = [None] * num_epochs
        U = [None] * num_epochs
        for epoch in range(num_epochs):
            print(epoch)
            X[epoch] = np.zeros((num_samples, H, W), dtype='float32')
            U[epoch] = np.zeros((num_samples, 2), dtype='float32')
            S[epoch] = np.zeros((num_samples, 4), dtype='float32')
            
            loop_trap = True
            while loop_trap:
                [s, x] = mdp.sample_random_state()
                i = 0
                loop_trap = False
                while (i < num_samples) and (not loop_trap):
                    cnt = 0
                    valid = False
                    while not valid:
                        cnt = cnt + 1
                        a = np.array([np.random.uniform(mdp.arange_min[0],
                                                        mdp.arange_max[0]),
                                      np.random.uniform(mdp.arange_min[1],
                                                        mdp.arange_max[1])])
                        [ns, nx] = mdp.transition_function(s, a)
                        valid = mdp.is_valid_state(ns)
                        if (cnt > 1000):
                            loop_trap = True
                            break
                        '''
                        print(s)
                        print(a)
                        print(ns)
                        print(valid)
                        print('#####################')
                        '''
                    a /= np.max(mdp.arange_max)
                    
        
                    S[epoch][i, :] = s
                    X[epoch][i, :, :] = x
                    U[epoch][i, :] = a
                    s = ns
                    x = nx
                    i = i+1
                    
        pickle.dump((X, U, S), open(filename, 'wb'))
        return X, U, S
    else:
        print('Loading plane dataset', filename)
        return pickle.load(open(filename, 'rb'))

get_moving_particle_dataset_trajectory('moving_particle_trajectory_train.data', num_samples = 400, num_epochs = 15)
get_moving_particle_dataset_trajectory('moving_particle_trajectory_test.data', num_samples = 400, num_epochs = 5)
      