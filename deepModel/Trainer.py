from deepModel.Logger import *
from deepModel.DeepNonLinearDynamicalSystem import *
import numpy as np
import keras
import keras.backend as K
from keras import optimizers
import tensorflow as tf

class Trainer:
    def __init__(self, deepNonLineerDynamicalSystem, save_history=False):
        self.deepNonLinearDynamicalSystem = deepNonLineerDynamicalSystem
        self.IterNum_EM = 10
        self.IterNum_CoordAsc = 5
        self.batch_size = 1000
        self.Ezt, self.EztztT, self.Ezt_1ztT = None, None, None
        self.w_all, self.v_all = None, None
        #self.iter_EM = -1
        #self.iter_CoorAsc = -1
        
        self.save_history = save_history
        if self.save_history==True:
            self.logger = Logger(self)
            self.hist_loss = {'observation_recons_loss':[], 'w_unit_norm_loss':[], 'w_LDS_loss':[], 'v_LDS_loss':[]}
            self.hist_loglik_w = []
            self.hist_EM_obj = []

#    def _sqr_diff(X):
#        tmp = K.tile(K.reshape(K.sum(K.square(X), axis=1), [-1,1]), [1,K.shape(X)[0]])
#        return tmp + K.transpose(tmp) - 2*tf.matmul(X, tf.transpose(X))
#
#    def _autoenc_reg_loss(x_true,w):
#        return K.mean(K.exp(-Trainer._sqr_diff(x_true)/kernel_sigma_2) * Trainer._sqr_diff(w), axis=-1) / K.mean(K.sum(K.square(w),axis=1))

    def _recons_loss(self, x_true, x_bar):
        return self.deepNonLinearDynamicalSystem.x_dim * keras.losses.mean_squared_error(x_true, x_bar) #keras.metrics.binary_crossentropy(x_true, x_bar)# might be better to be changed to binary_cross_entropy
    
    def _unit_norm_loss(self, fake_arg, x_bar):
        return K.square(tf.norm(x_bar, axis=1) - 1)


    def train_network(self, model, net_in, net_out, losses, lr, loss_weights, epochs, batch_size):
        model.compile(optimizer=optimizers.Adam(lr=lr,beta_1=0.1), loss=losses, loss_weights=loss_weights)
        model.fit( net_in, net_out, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=1)
                


    def train(self, x_all_train, u_all_train):#, x_all_validation, u_all_validation):
        
        x_train = np.concatenate(x_all_train)
        u_train = np.concatenate(u_all_train)
        #x_validation = np.concatenate(x_all_validation)
        #u_validation = np.concatenate(u_all_validation)
        
        for self.iter_EM in range(0,self.IterNum_EM):
            
            [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)
            
            [self.Ezt, self.EztztT, self.Ezt_1ztT] = self.deepNonLinearDynamicalSystem.kalmannModel.expectation(self.w_all,self.v_all)

            if self.iter_EM == 0:
                self.deepNonLinearDynamicalSystem.kalmannModel.maximization(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all)
                '''
                if self.save_history==True:
                    self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(Ezt, EztztT, Ezt_1ztT, self.w_all, self.v_all))
                    self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))
                    self.hist_loss['observation_recons_loss'].append()
                    self.hist_loss['w_unit_norm_loss'].append()
                '''                
            
            for self.iter_CoorAsc in range(self.IterNum_CoordAsc):
                
                ##### update observation_autoencoder parameters ###########################################
                w_LDS_loss = self._get_w_LDS_loss()
                
                #if (iter_EM == 0) and (iter_CoorAsc == 0):
                    #AE.load_weights('./cache_0_0_simpleAE_params.h5')
                
                EzT_CT_Rinv_plus_dT_Rinv = self._compute_EzT_CT_Rinv_plus_dT_Rinv()
                self.train_network(self.deepNonLinearDynamicalSystem.observation_autoencoder,\
                                   net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_plus_dT_Rinv],\
                                   losses = [self._recons_loss, self._unit_norm_loss, w_LDS_loss],\
                                   lr=0.005, loss_weights=[1., .1, 0.],
                                   epochs=200, batch_size=self.batch_size)
                self.train_network(self.DeepNonLinearDynamicalSystem.observation_autoencoder,\
                                   net_in=x_train, net_out=[x_train, x_train, EzT_CT_Rinv_plus_dT_Rinv],\
                                   losses = [self._recons_loss, self._unit_norm_loss, w_LDS_loss],\
                                   lr=0.005, loss_weights=[1., .1, 1.],
                                   epochs=1000, batch_size=self.batch_size)
                
                '''
                if self.save_history==True:
                    self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(Ezt, EztztT, Ezt_1ztT, self.w_all, self.v_all))
                    self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))
                    self.hist_loss['observation_recons_loss'].append()
                    self.hist_loss['w_unit_norm_loss'].append()
                '''
                
                ##########  update action_map parameters #############################                    
                
                v_LDS_loss = self._get_v_LDS_loss()
                
                EztT_minus_Ezt_1TAT_bT_alltimes_QinvH = self._compute_EztT_minus_Ezt_1TAT_bT_alltimes_QinvH()
                self.train_network(self.DeepNonLinearDynamicalSystem.action_map,\
                                   net_in=u_train, net_out=EztT_minus_Ezt_1TAT_bT_alltimes_QinvH,\
                                   losses = v_LDS_loss,\
                                   lr=0.00005, loss_weights=1,
                                   epochs=200, batch_size=self.batch_size)

                '''                
                if self.save_history==True:
                    self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(Ezt, EztztT, Ezt_1ztT, self.w_all, self.v_all))
                    self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))
                    self.hist_loss['observation_recons_loss'].append()
                    self.hist_loss['w_unit_norm_loss'].append()
                '''
                
                ############ update DLS parameters
                
                [self.w_all, self.v_all] = self.deepNonLinearDynamicalSystem.encode(x_all_train, u_all_train)
                
                self.deepNonLinearDynamicalSystem.kalmannModel.maximization(self.Ezt, self.EztztT, self.Ezt_1ztT, self.w_all, self.v_all)
                
                '''
                if self.save_history==True:
                    self.hist_EM_obj.append(self.deepNonLinearDynamicalSystem.kalmannModel.E_log_P_w_and_z(Ezt, EztztT, Ezt_1ztT, self.w_all, self.v_all))
                    self.hist_loglik_w.append(self.deepNonLinearDynamicalSystem.kalmannModel.log_likelihood(self.w_all, self.v_all))
                    self.hist_loss['observation_recons_loss'].append()
                    self.hist_loss['w_unit_norm_loss'].append()
                    #log_save_weights(iter_EM, iter_CoorAsc)        
                '''
            #log_save_weights(iter_EM, -1)

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