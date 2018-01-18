import datetime
import os
import pickle

class Logger:
    
    def __init__(self, trainer):
        self.trainer = trainer
        
        Logger._make_dir('./log')
        nw = datetime.datetime.now()
        self.base_log_dir = './log/' + str(nw.year) + '.' + str(nw.month) + '.' +\
                        str(nw.day) + '.' + str(nw.hour) + '.' + str(nw.minute) + '.' + str(nw.second) + '/'
        Logger._make_dir(self.base_log_dir)
 
    def _make_dir(dirname):           
        if not(os.path.isdir(dirname)):
            os.mkdir(dirname)

    def save_hist(self):
        Logger._make_dir(self.base_log_dir + str(self.trainer.iter_EM) + '/')
        log_dir = self.base_log_dir + str(self.trainer.iter_EM) + '/' + str(self.trainer.iter_CoorAsc) + '/'
        Logger._make_dir(log_dir)
        pickle.dump(self.trainer.hist_EM_obj, open(log_dir + 'hist_EM_obj.pkl', 'wb'))
        pickle.dump(self.trainer.hist_loglik_w, open(log_dir + 'hist_loglik_w.pkl', 'wb'))
        pickle.dump(self.trainer.hist_loss['observation_recons_loss'], open(log_dir + 'hist_observation_recons_loss.pkl', 'wb'))
        pickle.dump(self.trainer.hist_loss['w_regularization_loss'], open(log_dir + 'hist_w_regularization_loss.pkl', 'wb'))
        pickle.dump(self.trainer.hist_loss['w_LDS_loss'], open(log_dir + 'hist_w_LDS_loss.pkl', 'wb'))
        pickle.dump(self.trainer.hist_loss['v_LDS_loss'], open(log_dir + 'hist_v_LDS_loss.pkl', 'wb'))
        
    def save_params(self):
        Logger._make_dir(self.base_log_dir + str(self.trainer.iter_EM) + '/')
        log_dir = self.base_log_dir + str(self.trainer.iter_EM) + '/' + str(self.trainer.iter_CoorAsc) + '/'
        Logger._make_dir(log_dir)

        self.trainer.deepNonLinearDynamicalSystem.observation_autoencoder.save_weights(log_dir + 'observation_autoencoder_params.h5')
        self.trainer.deepNonLinearDynamicalSystem.action_encoder.save_weights(log_dir + 'action_encoder_params.h5')
        [A,b,H,C,d,Q,R,mu_0,Sig_0] = self.trainer.deepNonLinearDynamicalSystem.kalmannModel.getParams()
        pickle.dump([A,b,H,C,d,Q,R,mu_0,Sig_0], open(log_dir + 'LDS_params.pkl', 'wb'))