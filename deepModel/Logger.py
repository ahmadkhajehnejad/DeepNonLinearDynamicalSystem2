import datetime
import os

class Logger:
    
    def __init__(self, trainer):
        self.trainer = trainer
        
        Logger._make_dir('./log')
        nw = datetime.datetime.now()
        self.log_dir = './log/' + str(nw.year) + '.' + str(nw.month) + '.' +\
                        str(nw.day) + '.' + str(nw.hour) + '.' + str(nw.minute) + '.' + str(nw.second)
        Logger._make_dir(self.log_dir)
 
    def _make_dir(dirname):           
        if not(os.path.isdir(dirname)):
            os.mkdir(dirname)
        
        
'''    
    def print_EM_objective_function(self):
        [w_all, v_all] = self.trainer.deepNonLinearDynamicalSystem.
        for i in range(len(x_all)):
            w_all[i] = enc.predict(x_all[i])
        for i in range(len(u_all)):
            v_all[i] = act_map.predict(u_all[i])
        E_log.append(E_log_P_x_and_z(Ezt, EztztT, Ezt_1ztT,w_all,A,b,H,v_all,C,d,Q,R,mu_0,Sig_0))
        print('E[log...] = ', E_log)
'''