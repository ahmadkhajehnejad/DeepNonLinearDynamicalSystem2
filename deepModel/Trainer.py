from deepModel.Logger import *
from deepModel.DeepNonLinearDynamicalSystem import *
import numpy as np
import keras
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from deepModel.Logger import *

class Trainer:
    def __init__(self, deepNonLineerDynamicalSystem):
        self.deepNonLinearDynamicalSystem = deepNonLineerDynamicalSystem
        self.IterNum_EM = 10
        self.IterNum_CoordAsc = 5
        self.batch_size = 1000
        self.Ezt, self.EztztT, self.Ezt_1ztT = None, None, None
        self.w_all, self.v_all = None, None
        self.logger = Logger(self)
        
        self.hist_loss = {'observation_recons_loss':[], 'w_unit_norm_loss':[], 'w_LDS_loss':[], 'v_LDS_loss':[]}
        self.hist_loglik_w = []
        self.hist_EM_obj = []

    def _recons_loss(self, x_true, x_bar):
        return self.deepNonLinearDynamicalSystem.x_dim * keras.losses.mean_squared_error(x_true, x_bar) #keras.metrics.binary_crossentropy(x_true, x_bar)# might be better to be changed to binary_cross_entropy
    
    def _unit_norm_loss(self, fake_arg, x_bar):
        return 1600 * K.square(tf.norm(x_bar, axis=1) - 1)
        #return -K.log(tf.norm(x_bar, axis=1))
        #return -K.log(1 - K.square(2*(K.sigmoid(K.constant(np.array([10*(tf.norm(x_bar, axis=1)-1)]))) - 0.5)))

    '''
    def train_network(self, model, net_in, net_out, losses, lr, loss_weights, epochs, batch_size):
        model.compile(optimizer=optimizers.Adam(lr=lr,beta_1=0.1), loss=losses, loss_weights=loss_weights)
        h = model.fit( net_in, net_out, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=0,\
                      callbacks=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=20, \
                                                              verbose=1, mode='min')]).history.values()
        return list(h)
     '''
    
    def train_network(self, model, net_in, net_out, losses, lr, loss_weights, epochs, batch_size):
        model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
        h = model.fit( net_in, net_out, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=0,\
                      callbacks=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=20, \
                                                              verbose=1, mode='min')]).history.values()
        return list(h)         


    def train(self, x_all_train, u_all_train):#, x_all_validation, u_all_validation):
        
        x_train = np.concatenate(x_all_train)
        u_train = np.concatenate(u_all_train)
        #x_validation = np.concatenate(x_all_validation)
        #u_validation = np.concatenate(u_all_validation)

        [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)
        [self.Ezt, self.EztztT, self.Ezt_1ztT] = self.deepNonLinearDynamicalSystem.kalmannModel.expectation(self.w_all,self.v_all)
        self.deepNonLinearDynamicalSystem.kalmannModel.maximization(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all)
        
        
        for self.iter_EM in range(0,self.IterNum_EM):
            
            [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)            
            [self.Ezt, self.EztztT, self.Ezt_1ztT] = self.deepNonLinearDynamicalSystem.kalmannModel.expectation(self.w_all,self.v_all)
            
            if self.iter_EM == 0:
                self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all))
                self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))

            
            for self.iter_CoorAsc in range(self.IterNum_CoordAsc):
                ##### update observation_autoencoder parameters ###########################################

                w_LDS_loss = self._get_w_LDS_loss()
                
                #if (iter_EM == 0) and (iter_CoorAsc == 0):
                    #AE.load_weights('./cache_0_0_simpleAE_params.h5')
                
                EzT_CT_Rinv_plus_dT_Rinv = self._compute_EzT_CT_Rinv_plus_dT_Rinv()
                
                print('-------')
                current_w_LDS_loss = np.sum(K.eval(w_LDS_loss(K.constant(EzT_CT_Rinv_plus_dT_Rinv, dtype='float32'), K.constant(np.concatenate(self.w_all), dtype='float32'))))
                print(float(-current_w_LDS_loss))
                '''
                h_l = self.train_network(self.deepNonLinearDynamicalSystem.observation_autoencoder,\
                                   net_in=x_train, net_out=[x_train, x_train,EzT_CT_Rinv_plus_dT_Rinv],\
                                   losses = [self._recons_loss, self._unit_norm_loss, w_LDS_loss],\
                                   lr=0.00005, loss_weights=[1., 0., .1],
                                   epochs=200, batch_size=self.batch_size)
                '''
                h_l = self.train_network(self.deepNonLinearDynamicalSystem.observation_autoencoder,\
                                   net_in=x_train, net_out=[x_train, x_train,EzT_CT_Rinv_plus_dT_Rinv],\
                                   losses = [self._recons_loss, self._unit_norm_loss, w_LDS_loss],\
                                   lr=0.00000005, loss_weights=[10., 0., 1.],
                                   epochs=200, batch_size=self.batch_size)
                
                [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)
                self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all))
                self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))
                
                self.hist_loss['observation_recons_loss'].append(h_l[1])
                self.hist_loss['w_unit_norm_loss'].append(h_l[2])
                self.hist_loss['w_LDS_loss'].append(h_l[3])

                current_w_LDS_loss = np.sum(K.eval(w_LDS_loss(K.constant(EzT_CT_Rinv_plus_dT_Rinv, dtype='float32'), K.constant(np.concatenate(self.w_all), dtype='float32'))))
                print(float(-current_w_LDS_loss))
                k_current_w_LDS_loss = self.deepNonLinearDynamicalSystem.kalmannModel.get_w_LDS_loss(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all)
                print(k_current_w_LDS_loss)
                k_current_w_LDS_loss_2 = self.deepNonLinearDynamicalSystem.kalmannModel.get_w_LDS_loss_2(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all)
                print(k_current_w_LDS_loss_2)
                print('-------')
                print()
                

                ##########  update action_map parameters #############################                    
                
                v_LDS_loss = self._get_v_LDS_loss()
                
                EztT_minus_Ezt_1TAT_bT_alltimes_QinvH = self._compute_EztT_minus_Ezt_1TAT_bT_alltimes_QinvH()
                h_l = self.train_network(self.deepNonLinearDynamicalSystem.action_encoder,\
                                   net_in=u_train, net_out=EztT_minus_Ezt_1TAT_bT_alltimes_QinvH,\
                                   losses = v_LDS_loss,\
                                   lr=0.00005, loss_weights=[1],
                                   epochs=200, batch_size=self.batch_size)

                [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)
                self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all))
                self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))
                
                self.hist_loss['v_LDS_loss'].append(h_l[0])
                
                
                ############ update DLS parameters
                
                self.deepNonLinearDynamicalSystem.kalmannModel.maximization(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all)

                [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)                                
                self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all))
                self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))

                print('EM_objective : ' + str(self.hist_EM_obj))
                #print('loglik_w : ' + str(self.hist_loglik_w))
                print()
                print('reoncs,  union,  LDS =')
                print([self.hist_loss['observation_recons_loss'][-1][-1], np.exp(-self.hist_loss['w_unit_norm_loss'][-1][-1]), self.hist_loss['w_LDS_loss'][-1][-1]])
                print()

                self.logger.save_hist()
                self.logger.save_params()

    def _compute_EzT_CT_Rinv_plus_dT_Rinv(self):
        w_dim = self.deepNonLinearDynamicalSystem.w_dim
        [_,_,_,C,d,_,R,_,_] = self.deepNonLinearDynamicalSystem.kalmannModel.getParams()
        Rinv = np.linalg.inv(R)
        n_ = sum(a.shape[0] for a in self.w_all)
        EzT_CT_Rinv_plus_dT_Rinv = np.zeros([n_,w_dim])
        i_start, i_end = 0, -1
        for i in range(len(self.w_all)):
            i_end = i_start + self.w_all[i].shape[0]
            EzT_CT = np.matmul(self.Ezt[i].T, C.T)
            EzT_CT_plus_dT = EzT_CT + np.tile(d.reshape([1,-1]),[EzT_CT.shape[0],1])
            EzT_CT_Rinv_plus_dT_Rinv[i_start:i_end,:] = np.matmul(EzT_CT_plus_dT, Rinv)
            i_start = i_end
        return EzT_CT_Rinv_plus_dT_Rinv 

    def _get_w_LDS_loss(self):
        [_,_,_,_,_,_,R,_,_] = self.deepNonLinearDynamicalSystem.kalmannModel.getParams()
        Rinv = np.linalg.inv(R)
        Rinv_tf = tf.constant(Rinv, dtype='float32')
        def w_LDS_loss(EzT_CT_Rinv_plus_dT_Rinv, w):
            sh = K.shape(w)
            return -tf.matmul(tf.reshape(EzT_CT_Rinv_plus_dT_Rinv,[sh[0],1,sh[1]]), tf.reshape(w,[sh[0],sh[1],1])) \
                        + 0.5 * tf.matmul(\
                                          tf.reshape(tf.matmul(w,Rinv_tf),[sh[0],1,-1])\
                                          ,tf.reshape(w,[sh[0],sh[1],1])\
                                         )
        return w_LDS_loss

    def _compute_EztT_minus_Ezt_1TAT_bT_alltimes_QinvH(self):
        v_dim = self.deepNonLinearDynamicalSystem.v_dim
        [A,b,H,_,_,Q,_,_,_] = self.deepNonLinearDynamicalSystem.kalmannModel.getParams()
        Qinv = np.linalg.inv(Q)
        n_ = sum(a.shape[0] for a in self.v_all)
        EztT_minus_Ezt_1TAT_bT_alltimes_QinvH = np.zeros([n_,v_dim])
        i_start, i_end = 0, -1
        for i in range(len(self.v_all)):
            i_end = i_start + self.v_all[i].shape[0]
            EztT_minus_Ezt_1TAT_bT = self.Ezt[i][:,1:].T - np.matmul(self.Ezt[i][:,:-1].T,A.T) - np.tile(b.T,[self.Ezt[i].shape[1]-1,1])
            EztT_minus_Ezt_1TAT_bT_alltimes_QinvH[i_start:i_end,:] = np.matmul(np.matmul(EztT_minus_Ezt_1TAT_bT, Qinv),H)
            i_start = i_end
        return EztT_minus_Ezt_1TAT_bT_alltimes_QinvH

    def _get_v_LDS_loss(self):
        [_,_,H,_,_,Q,_,_,_] = self.deepNonLinearDynamicalSystem.kalmannModel.getParams()
        Qinv = np.linalg.inv(Q)
        HTQinvH = np.matmul(np.matmul(H.T, Qinv),H)
        HTQinvH_tf = tf.constant(HTQinvH, dtype='float32')
        def v_LDS_loss(EztT_minus_Ezt_1TAT_bT_alltimes_QinvH, v):
            sh = K.shape(v)
            return -tf.matmul(tf.reshape(EztT_minus_Ezt_1TAT_bT_alltimes_QinvH, [sh[0],1,sh[1]]), tf.reshape(v,[sh[0],sh[1],1]))\
            + 0.5 * tf.matmul(tf.reshape(tf.matmul(v, HTQinvH_tf),[sh[0],1,sh[1]]), tf.reshape(v,[sh[0],sh[1],1]))
        return v_LDS_loss
