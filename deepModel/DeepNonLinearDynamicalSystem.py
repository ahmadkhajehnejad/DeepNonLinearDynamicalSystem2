import numpy as np
from keras.models import Model , Sequential
from keras.layers import Dense, Input, Reshape, Lambda, Concatenate
from keras import backend as K
import tensorflow as tf
from KalmannModel.KalmannModel import *
from deepModel.Trainer import *

class DeepNonLinearDynamicalSystem:
    def __init__(self):
        self.w_dim, self.z_dim, self.v_dim, self.x_dim, self.u_dim = 10, 4, 4, 40*40, 2
        self.intermediate_dim = 500
        self.kalmannModel = KalmannModel(self.z_dim, self.w_dim, self.v_dim)
        
        x = Input(shape=(self.x_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        w = Dense(self.w_dim, activation=None)(h)
        
        self.observation_encoder = Model([x],[w])
        
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.x_dim, activation='sigmoid')
        h_decoded = decoder_h(w)
        x_bar = decoder_mean(h_decoded)
        
        #decoder_input = Input(shape=(self.w_dim,))
        #_h_decoded = decoder_h(decoder_input)
        #_x_decoded_mean = decoder_mean(_h_decoded)
        #dec = Model(decoder_input, _x_decoded_mean)
        
        self.observation_autoencoder = Model(x,[x_bar,w,w])
        #self.observation_autoencoder = Model(x,x_bar)
        
        self.action_encoder = Sequential()
        self.action_encoder.add(Dense(4, input_shape=(self.u_dim,), activation='relu'))
        self.action_encoder.add(Dense(self.v_dim, activation=None))
        
        self.trainer = Trainer(self)
        
    def encode(self, x_all, u_all):
        w_all = [self.observation_encoder.predict(a) for a in x_all]
        v_all = [self.action_encoder.predict(a) for a in u_all]
        return [w_all, v_all]
                
    def train(self, x_all_train, u_all_train): 
        self.trainer.train(x_all_train, u_all_train)
    
