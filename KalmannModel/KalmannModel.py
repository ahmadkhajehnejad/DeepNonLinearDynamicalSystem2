import numpy as np
from pykalman import KalmanFilter
from scipy.stats import multivariate_normal

class KalmannModel:
    
    def __init__(self, state_dim, observation_dim, action_dim):
        self.mu_0, self.Sig_0 = np.zeros([state_dim,1]), np.eye(state_dim)
        self.A = np.eye(state_dim) + np.random.uniform(-0.1,0.1,state_dim*state_dim).reshape([state_dim,state_dim])
        self.b = np.zeros([state_dim,1])
        self.H = np.ones([state_dim, action_dim])/action_dim
        self.Q = np.eye(state_dim)
        self.C = np.ones([observation_dim, state_dim])/state_dim + np.random.uniform(-0.1,0.1,observation_dim*state_dim).reshape([observation_dim,state_dim])
        self.d,self.R = np.zeros([observation_dim,1]), np.eye(observation_dim)
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    '''
    def ch_inv(X):
        X_ch = np.linalg.inv(np.linalg.cholesky(X))
        X_inv = np.dot(X_ch.T,X_ch)
        return X_inv
    '''
    
    def expectation(self,w_all,v_all):
        
        M = len(w_all)
        T = w_all[0].shape[0]
        
        Ezt = [None] * M
        EztztT = [None] * M
        Ezt_1ztT = [None] * M
        Sigt = [None] * M
        Lt = [None] * M
        
        z_dim = self.state_dim
        for i in range(M):
           
            Ezt[i] = np.zeros([z_dim,T])
            EztztT[i] = [None] * T
            Ezt_1ztT[i] = [None] * T
            Sigt[i] = [None] * T
            Lt[i] = [None] * T
                
            
            mu_t_t = [None] * T
            Sig_t_t = [None] * T
            mu_t_t_1 = [None] * T
            Sig_t_t_1 = [None] * T
            mu_t_T = [None] * T
            Sig_t_T = [None] * T
            
            for t in range(T):
                if t > 0:
                    mu_t_t_1[t] = np.matmul(self.A,mu_t_t[t-1]) + np.matmul(self.H,v_all[i][t-1,:].reshape([-1,1])) + self.b
                    Sig_t_t_1[t] = np.matmul(self.A, np.matmul(Sig_t_t[t-1],self.A.T)) + self.Q
                else:
                    mu_t_t_1[t] = self.mu_0
                    Sig_t_t_1[t] = self.Sig_0
                    
                ''' 
                K = np.linalg.solve(\
                                    (np.matmul(\
                                               self.C,\
                                               np.matmul(Sig_t_t_1[t], self.C.T)\
                                               ) + self.R\
                                    ).T\
                                    ,np.matmul(Sig_t_t_1[t], self.C.T).T\
                                   ).T
                '''
                K = np.matmul(\
                              np.matmul(Sig_t_t_1[t], self.C.T) , \
                              np.linalg.inv(\
                                            np.matmul(\
                                                      self.C,
                                                      np.matmul(Sig_t_t_1[t], self.C.T)\
                                                     ) + self.R\
                                           )\
                             )
                
                mu_t_t[t] = mu_t_t_1[t] + np.matmul(K,w_all[i][t,:].reshape([-1,1]) - np.matmul(self.C,mu_t_t_1[t]) - self.d)
                Sig_t_t[t] = np.matmul(np.eye(K.shape[0]) - np.matmul(K,self.C), Sig_t_t_1[t])
                
            for t in reversed(range(T)):
                if t == (T - 1):
                    mu_t_T[t] = mu_t_t[t]
                    Sig_t_T[t] = Sig_t_t[t]
                else:
                    
                    #L = np.linalg.solve(Sig_t_t_1[t+1].T, np.matmul(Sig_t_t[t], self.A.T).T).T
                    L = np.matmul(Sig_t_t[t], np.matmul(self.A.T, np.linalg.inv(Sig_t_t_1[t+1])))
                    mu_t_T[t] = mu_t_t[t] + np.matmul(L,mu_t_T[t+1] - mu_t_t_1[t+1])
                    Sig_t_T[t] = Sig_t_t[t] + np.matmul(\
                                                        L,\
                                                        np.matmul(\
                                                                  (Sig_t_T[t+1] - Sig_t_t_1[t+1]),\
                                                                  L.T\
                                                                 )\
                                                       )
                Ezt[i][:,t] = mu_t_T[t].reshape([-1])
                EztztT[i][t] = Sig_t_T[t] + np.matmul(mu_t_T[t], mu_t_T[t].T)
                if t < (T - 1):
                    Ezt_1ztT[i][t+1] = np.matmul(mu_t_t[t], mu_t_T[t+1].T) +\
                                     np.matmul(L,EztztT[i][t+1]) -\
                                     np.matmul(L,np.matmul(mu_t_t_1[t+1], mu_t_T[t+1].T))
                Sigt[i][t] = Sig_t_T[t]
                if t < (T-1):
                    Lt[i][t] = L
        
        return [Ezt, EztztT, Ezt_1ztT]
    
    
    
    def maximization(self, Ezt, EztztT, Ezt_1ztT, w_all, v_all):
        
        w_dim = self.observation_dim
        v_dim = self.action_dim
        z_dim = self.state_dim
        
        M = len(Ezt)
        T = Ezt[0].shape[1]
        
        b_old = self.b
        d_old = self.d
        
        ############################ mu_0
        
        tmp = np.zeros([z_dim,1])
        for i in range(M):
            tmp = tmp + Ezt[i][:,0].reshape([-1,1])
        self.mu_0 = tmp / M
        
        ############################ Sigma_0
        
        tmp = np.zeros([z_dim,z_dim])
        for i in range(M):
            tmp = tmp + EztztT[i][0]
        self.Sig_0 = (tmp / M) - np.matmul(self.mu_0, self.mu_0.T)
    
        ############################ H
        
        #self.H = np.ones([z_dim, v_dim]) / v_dim
        #self.H = np.eye(z_dim)
        self.H = np.zeros(z_dim)

        
        ############################ A
        
        mean_Ezt = np.zeros([z_dim,1])
        mean_Ezt_1 = np.zeros([z_dim,1])
        mean_Ezt_1ztT = np.zeros([z_dim,z_dim])
        mean_Ezt_1zt_1T = np.zeros([z_dim,z_dim])
        mean_Hvt_1 = np.zeros([z_dim,1])
        mean_Hvt_1Ezt_1T = np.zeros([z_dim,z_dim])
        
        for i in range(M):
            for t in range(1,T):
                mean_Ezt = mean_Ezt + Ezt[i][:,t].reshape([-1,1])
                mean_Ezt_1 = mean_Ezt_1 + Ezt[i][:,t-1].reshape([-1,1])
                mean_Ezt_1ztT = mean_Ezt_1ztT + Ezt_1ztT[i][t]
                mean_Ezt_1zt_1T = mean_Ezt_1zt_1T + EztztT[i][t-1]
                mean_Hvt_1 = mean_Hvt_1 + np.matmul(self.H,v_all[i][t-1,:].reshape([-1,1]))
                mean_Hvt_1Ezt_1T = mean_Hvt_1Ezt_1T + np.matmul(np.matmul(self.H,v_all[i][t-1,:].reshape([-1,1])), Ezt[i][:,t-1].reshape([1,-1]))
          
        if (T > 1):
            mean_Ezt = mean_Ezt / ((T-1)*M)
            mean_Ezt_1 = mean_Ezt_1 / ((T-1)*M)
            mean_Ezt_1ztT = mean_Ezt_1ztT / ((T-1)*M)
            mean_Ezt_1zt_1T = mean_Ezt_1zt_1T / ((T-1)*M)
            mean_Hvt_1 = mean_Hvt_1 / ((T-1)*M)
            mean_Hvt_1Ezt_1T = mean_Hvt_1Ezt_1T / ((T-1)*M)
        
        #tmp_1 = mean_Ezt_1ztT.T - np.matmul(mean_Ezt, mean_Ezt_1.T) - mean_Hvt_1Ezt_1T + np.matmul(mean_Hvt_1, mean_Ezt_1.T)
        #tmp_2 = mean_Ezt_1zt_1T - np.matmul(mean_Ezt_1, mean_Ezt_1.T)
        tmp_1 = mean_Ezt_1ztT.T - np.matmul(b_old, mean_Ezt_1.T) - np.matmul(mean_Hvt_1, mean_Ezt_1.T)
        tmp_2 = mean_Ezt_1zt_1T
        
        self.A = np.matmul(tmp_1, np.linalg.inv(tmp_2))
        #self.A = np.matmul(tmp_1, np.linalg.pinv(tmp_2))
        #self.A = np.linalg.solve(tmp_2.T,tmp_1.T).T
    
        ############################ b
        
        self.b = mean_Ezt - np.matmul(self.A,mean_Ezt_1) - mean_Hvt_1
        
        ############################ Q
        
        tmp = np.zeros([z_dim,z_dim])
        for i in range(M):
            for t in range(1,T):
                
                
                #tmp_0 = np.matmul(self.H,v_all[i][t-1,:].reshape([-1,1])).reshape([-1,1]) + self.b
                tmp_0 = np.matmul(self.H,v_all[i][t-1,:].reshape([-1,1])).reshape([-1,1]) + b_old
                
                tmp = tmp + EztztT[i][t] - np.matmul(Ezt_1ztT[i][t].T, self.A.T) - np.matmul(self.A,Ezt_1ztT[i][t])\
                          - np.matmul(tmp_0, Ezt[i][:,t].reshape([1,-1])) - np.matmul(Ezt[i][:,t].reshape([-1,1]), tmp_0.T)\
                          + np.matmul(self.A, np.matmul(EztztT[i][t-1], self.A.T)) + np.matmul(tmp_0,np.matmul(Ezt[i][:,t-1].reshape([1,-1]), self.A.T).reshape([1,-1]))\
                          + np.matmul(self.A, np.matmul(Ezt[i][:,t-1].reshape([-1,1]), tmp_0.T)) + np.matmul(tmp_0, tmp_0.T)
    
        self.Q = tmp / ((T-1)*M)
        
        ############################ C
        mean_wt = np.zeros([w_dim,1])
        mean_wtEztT = np.zeros([w_dim, z_dim])
        mean_Ezt = np.zeros([z_dim, 1])
        mean_EztztT = np.zeros([z_dim,z_dim])
        for i in range(M):
            for t in range(T):
                mean_wt = mean_wt + w_all[i][t,:].reshape([-1,1])
                mean_wtEztT = mean_wtEztT + np.matmul(w_all[i][t,:].reshape([-1,1]),Ezt[i][:,t].reshape([1,-1]))
                mean_Ezt = mean_Ezt + Ezt[i][:,t].reshape([-1,1])
                mean_EztztT = mean_EztztT + EztztT[i][t]
        mean_wt = mean_wt / (T*M)
        mean_wtEztT = mean_wtEztT / (T*M)
        mean_Ezt = mean_Ezt / (T*M)
        mean_EztztT = mean_EztztT / (T*M)
        
        #tmp_1 = mean_wtEztT - np.matmul(mean_wt, mean_Ezt.T)
        #tmp_2 = mean_EztztT - np.matmul(mean_Ezt, mean_Ezt.T)
        tmp_1 = mean_wtEztT - np.matmul(d_old, mean_Ezt.T)
        tmp_2 = mean_EztztT    
        
        self.C = np.matmul(tmp_1, np.linalg.inv(tmp_2))
        #C = np.matmul(tmp_1, np.linalg.pinv(tmp_2))
        #C = np.linalg.solve(tmp_2.T,tmp_1.T).T
        ############################ d
        
        self.d = mean_wt - np.matmul(self.C,mean_Ezt)
        
        ############################ R
        tmp = np.zeros([w_dim,w_dim])
        for i in range(M):
            for t in range(T):
                #tmp_0 = w_all[i][t,:].reshape([-1,1]) - self.d
                tmp_0 = w_all[i][t,:].reshape([-1,1]) - d_old
                tmp = tmp + np.matmul(tmp_0, tmp_0.T)\
                          + np.matmul(self.C,np.matmul(EztztT[i][t],self.C.T))\
                          - np.matmul(tmp_0, np.matmul(Ezt[i][:,t].reshape([1,-1]),self.C.T).reshape([1,-1]))\
                          - np.matmul(np.matmul(self.C,Ezt[i][:,t].reshape([-1,1])).reshape([-1,1]), tmp_0.T)
        self.R = tmp / (T*M)    
    
    
    def EM_step(self,w_all,v_all):
        [Ezt, EztztT, Ezt_1ztT] = self.expectation(w_all,v_all)
        self.maximization(Ezt, EztztT, Ezt_1ztT, w_all, v_all)

    def getParams(self):
        return [self.A,self.b,self.H,self.C,self.d,self.Q,self.R,self.mu_0,self.Sig_0]
    
    def setParams(self, A, b, H, C, d, Q, R, mu_0, Sig_0):
        self.A, self.b, self.H, self.C = A, b, H, C
        self.d, self.Q, self.R, self.mu_0, self.Sig_0 = d, Q, R, mu_0, Sig_0
        
    def log_likelihood(self,w_all,v_all):
        
        M = len(w_all)
        T = w_all[0].shape[0]
        
        Ezt = [None] * M
        EztztT = [None] * M
        Ezt_1ztT = [None] * M
        
        z_dim = self.state_dim
        L = 0
        for i in range(M):
           
            Ezt[i] = np.zeros([z_dim,T])
            EztztT[i] = [None] * T
            Ezt_1ztT[i] = [None] * T            
            
            
            mu_t_t = [None] * T
            Sig_t_t = [None] * T
            mu_t_t_1 = [None] * T
            Sig_t_t_1 = [None] * T
            mu_t_T = [None] * T
            Sig_t_T = [None] * T
            
            for t in range(T):
                if t > 0:
                    mu_t_t_1[t] = np.matmul(self.A,mu_t_t[t-1]) + np.matmul(self.H,v_all[i][t-1,:].reshape([-1,1])) + self.b
                    Sig_t_t_1[t] = np.matmul(self.A, np.matmul(Sig_t_t[t-1],self.A.T)) + self.Q
                else:
                    mu_t_t_1[t] = self.mu_0
                    Sig_t_t_1[t] = self.Sig_0
                    
                ''' 
                K = np.linalg.solve(\
                                    (np.matmul(\
                                               self.C,\
                                               np.matmul(Sig_t_t_1[t], self.C.T)\
                                               ) + self.R\
                                    ).T\
                                    ,np.matmul(Sig_t_t_1[t], self.C.T).T\
                                   ).T
                '''
                K = np.matmul(\
                              np.matmul(Sig_t_t_1[t], self.C.T) , \
                              np.linalg.inv(\
                                            np.matmul(\
                                                      self.C,
                                                      np.matmul(Sig_t_t_1[t], self.C.T)\
                                                     ) + self.R\
                                           )\
                             )
                
                mu_t_t[t] = mu_t_t_1[t] + np.matmul(K,w_all[i][t,:].reshape([-1,1]) - np.matmul(self.C,mu_t_t_1[t]) - self.d)
                Sig_t_t[t] = np.matmul(np.eye(K.shape[0]) - np.matmul(K,self.C), Sig_t_t_1[t])
                mu_xt_x1tot_1 = np.matmul(self.C,mu_t_t_1[t]) + self.d
                Sig_xt_x1tot_1 = np.matmul(np.matmul(self.C,Sig_t_t_1[t]),self.C.T) + self.R
                
                L = L - 0.5*np.log(2*np.pi*np.linalg.det(Sig_xt_x1tot_1)) -\
                    0.5 * np.matmul(np.matmul(\
                                              w_all[i][t,:].reshape([1,-1]) - mu_xt_x1tot_1.T,\
                                              np.linalg.inv(Sig_xt_x1tot_1)\
                                             ),\
                                    w_all[i][t,:].reshape([-1,1]) - mu_xt_x1tot_1)
                
                
                #L = L + multivariate_normal.logpdf(w_all[i][t,:],mean=mu_xt_x1tot_1.reshape([-1]),cov=Sig_xt_x1tot_1)
            
        return L[0][0]
    
    def E_log_P_w_and_z(self, Ezt, EztztT, Ezt_1ztT, w_all, v_all): ## Expectations have been computed with \theta^{old}
        E = 0
        M = len(Ezt)
        T = Ezt[0].shape[1]
        Sig_0_inv = np.linalg.inv(self.Sig_0)
        Q_inv = np.linalg.inv(self.Q)
        R_inv = np.linalg.inv(self.R)
        for m in range(M):
            E = E - 0.5 * np.log(2 * np.pi * np.linalg.det(self.Sig_0))
            E = E - 0.5 * np.trace(np.matmul(EztztT[m][0], Sig_0_inv))
            E = E + np.matmul(np.matmul(self.mu_0.T, Sig_0_inv).reshape([1,-1]), Ezt[m][:,0].reshape([-1,1]))
            E = E - 0.5 * np.matmul(np.matmul(self.mu_0.T, Sig_0_inv).reshape([1,-1]), self.mu_0)
            for t in range(1,T):
                E = E - 0.5 * np.log(2 * np.pi * np.linalg.det(self.Q))
                E = E - 0.5 * np.trace(np.matmul(EztztT[m][t],Q_inv))
                E = E + np.trace(np.matmul(np.matmul(self.A,Ezt_1ztT[m][t]),Q_inv))
                Hv_t_1_plus_b = np.matmul(self.H,v_all[m][t-1,:].reshape([-1,1])).reshape([-1,1])+self.b
                E = E + np.matmul(Hv_t_1_plus_b.T,np.matmul(Q_inv,Ezt[m][:,t].reshape([-1,1])).reshape([-1,1]))
                E = E - 0.5 * np.trace(np.matmul(np.matmul(EztztT[m][t-1],self.A.T),np.matmul(Q_inv,self.A)))
                E = E - np.matmul(np.matmul((Hv_t_1_plus_b).T,Q_inv),np.matmul(self.A,Ezt[m][:,t-1].reshape([-1,1])))
                E = E - 0.5 * np.matmul(np.matmul(Hv_t_1_plus_b.T,Q_inv).reshape([1,-1]), Hv_t_1_plus_b)
            for t in range(T):
                E = E - 0.5 * np.log(2 * np.pi * np.linalg.det(self.R))
                w_minus_d = w_all[m][t,:].reshape([-1,1]) - self.d
                E = E - 0.5 * np.matmul(np.matmul(w_minus_d.T,R_inv).reshape([1,-1]), w_minus_d)
                E = E - 0.5 * np.trace(np.matmul(np.matmul(EztztT[m][t],self.C.T),np.matmul(R_inv,self.C)))
                E = E + np.matmul(np.matmul(w_minus_d.T,R_inv).reshape([1,-1]), np.matmul(self.C,Ezt[m][:,t].reshape([-1,1])).reshape([-1,1]))
        return E[0][0]
    
    def get_w_LDS_loss(self, Ezt, EztztT, Ezt_1ztT, w_all):
        E = 0
        M = len(Ezt)
        T = Ezt[0].shape[1]
        Sig_0_inv = np.linalg.inv(self.Sig_0)
        Q_inv = np.linalg.inv(self.Q)
        R_inv = np.linalg.inv(self.R)
        for m in range(M):
            for t in range(T):
                w_minus_d = w_all[m][t,:].reshape([-1,1]) - self.d
                E = E - 0.5 * np.matmul(np.matmul(w_minus_d.T,R_inv).reshape([1,-1]), w_minus_d)
                E = E + 0.5 * np.matmul(np.matmul(self.d.reshape([1,-1]), R_inv).reshape([1,-1]), self.d.reshape([-1,1]))
                E = E + np.matmul(np.matmul(w_all[m][t,:].reshape([1,-1]),R_inv).reshape([1,-1]), np.matmul(self.C,Ezt[m][:,t].reshape([-1,1])).reshape([-1,1]))
        return E[0][0]
    def get_w_LDS_loss_2(self, Ezt, EztztT, Ezt_1ztT, w_all):
        E = 0
        M = len(Ezt)
        T = Ezt[0].shape[1]
        Sig_0_inv = np.linalg.inv(self.Sig_0)
        Q_inv = np.linalg.inv(self.Q)
        R_inv = np.linalg.inv(self.R)
        for m in range(M):
            for t in range(T):
                w_minus_d = w_all[m][t,:].reshape([-1,1]) - self.d
                E = E - 0.5 * np.matmul(np.matmul(w_all[m][t,:].reshape([1,-1]),R_inv).reshape([1,-1]), w_all[m][t,:].reshape([-1,1]))
                E = E + np.matmul(w_all[m][t,:].reshape([1,-1]),np.matmul(R_inv, np.matmul(self.C,Ezt[m][:,t].reshape([-1,1])).reshape([-1,1]) + self.d.reshape([-1,1])))
        return E[0][0]

    
    def predict(self,w,v):
        z_dim = self.state_dim
        w_dim = self.observation_dim
        w_all = [w]
        v_all = [v[:w.shape[0]]]
        [Ezt, _, _] = self.expectation(w_all,v_all)
        mu_z = np.zeros([v.shape[0]-w.shape[0]+1, z_dim])
        est = np.zeros([v.shape[0]-w.shape[0]+1, w_dim])
        k = 0
        for i in range(w.shape[0],v.shape[0]+1):
            if k == 0:
                mu_z[k,:] = np.matmul(Ezt[0][:,-1].reshape([1,-1]),self.A.T) + np.matmul(v[i-1,:].reshape([1,-1]),self.H.T) + self.b.reshape([-1])
            else:
                mu_z[k,:] = np.matmul(mu_z[k-1,:].reshape([1,-1]),self.A.T) + np.matmul(v[i-1,:].reshape([1,-1]),self.H.T) + self.b.reshape([-1])
            est[k] = np.matmul(mu_z[k,:].reshape([1,-1]),self.C.T) + self.d.T
            k = k+1
        return est
    
def _test_KalmanModel():
    
    w_dim, z_dim, v_dim = 4, 3, 5
    EM_iter = 10
    
    T = 5
    M = 1
    w_all = [None]*M
    v_all = [None]*M
    for i in range(M):
        for t in range(T):
            w_all[i] = np.random.uniform(-1,1,w_dim*T).reshape([T,w_dim])
            v_all[i] = np.zeros([T,v_dim])
            
    #mu_0, Sig_0 = np.zeros([z_dim,1]), np.eye(z_dim)
    #A,b,H,Q = np.eye(z_dim), np.zeros([z_dim,1]), np.ones([z_dim, v_dim])/v_dim, np.eye(z_dim)
    #C,d,R = np.ones([w_dim, z_dim])/z_dim + np.random.uniform(-0.1,0.1,w_dim*z_dim).reshape([w_dim,z_dim]), np.zeros([w_dim,1]), np.eye(w_dim)
    
    km = KalmannModel(state_dim=z_dim, observation_dim=w_dim, action_dim=v_dim)
    [A,b,H,C,d,Q,R,mu_0,Sig_0] = km.getParams()
    
    kf = KalmanFilter(initial_state_mean = mu_0.reshape([-1]),
                      initial_state_covariance = Sig_0,
                      transition_matrices = A,
                      transition_offsets = b.reshape([-1]),
                      transition_covariance = Q,
                      observation_matrices = C,
                      observation_offsets = d.reshape([-1]),
                      observation_covariance = R)
    
    kf = kf.em(w_all[0],n_iter=EM_iter,em_vars=['initial_state_mean', 'initial_state_covariance',\
               'transition_matrices', 'transition_offsets', 'transition_covariance', \
               'observation_matrices', 'observation_offsets', 'observation_covariance'])
    
    
    for i in range(EM_iter):
        km.EM_step(w_all,v_all)
    [A,b,H,C,d,Q,R,mu_0,Sig_0] = km.getParams()
        
    
    #print(mu_0.reshape([-1]))
    #print(kf.initial_state_mean)
    print(np.max(np.abs(mu_0.reshape([-1]) - kf.initial_state_mean)))
    print()
    
    #print(Sig_0)
    #print(kf.initial_state_covariance)
    print(np.max(np.abs(Sig_0 - kf.initial_state_covariance)))
    print()
    
    #print(A)
    #print(kf.transition_matrices)
    print(np.max(np.abs(A - kf.transition_matrices)))
    print()
    
    #print(b.reshape([-1]))
    #print(kf.transition_offsets)
    print(np.max(np.abs(b.reshape([-1]) - kf.transition_offsets)))
    print()
    
    #print(Q)
    #print(kf.transition_covariance)
    print(np.max(np.abs(Q - kf.transition_covariance)))
    print()
    
    print("----------------")
    print()
    
    #print(C)
    #print(kf.observation_matrices)
    print(np.max(np.abs(C - kf.observation_matrices)))
    print()
    
    
    #print(d.reshape([-1]).)
    #print(kf.observation_offsets)
    print(np.max(np.abs(d.reshape([-1]) - kf.observation_offsets)))
    print()
    
    #print(R)
    #print(kf.observation_covariance)
    print(np.max(np.abs(R - kf.observation_covariance)))
    print()


    
    L1 = km.log_likelihood(w_all,v_all)
    
    [_,_,H,_,_,_,_,_,_]
    km.setParams(kf.transition_matrices, kf.transition_offsets.reshape([-1,1]), H,\
                 kf.observation_matrices, kf.observation_offsets.reshape([-1,1]),\
                 kf.transition_covariance, kf.observation_covariance,\
                 kf.initial_state_mean.reshape([-1,1]), kf.initial_state_covariance)
    L2 = km.log_likelihood(w_all,v_all)
                        
    print(L1)
    print(L2)
     