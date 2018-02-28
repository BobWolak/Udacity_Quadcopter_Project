import numpy as np
import math
from keras import Sequential
from keras import layers, models, optimizers
from keras.regularizers import l2
from keras import backend as K

class Critic_Hover:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size= action_size
        
        
        self.build_model()
        
    def build_model(self):
        states=layers.Input(shape=(self.state_size,), name='states')
        actions=layers.Input(shape=(self.action_size,),name='actions')
        
        h1_states=layers.Dense(units=64, activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(states)
        h1_states=layers.Dropout(rate=0.2)(h1_states)
        h2_states= layers.Dense(units=64, activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(h1_states)
        h2_states =layers.BatchNormalization()(h2_states)
        
        h1_actions=layers.Dense(units=64, activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(actions)
        h1_actions=layers.Dropout(rate=0.2)(h1_actions)
        h2_actions = layers.Dense(units=64, activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(h1_actions)
        h2_actions = layers.BatchNormalization()(h2_actions)
        
        net=layers.Add()([h2_states,h2_actions])
        net=layers.Activation('relu')(net)
        
        Q_values=layers.Dense(units=1,name='Q_values')(net)
        
        self.model = models.Model(inputs=[states,actions], output=Q_values)
        
        optimizer = optimizers.Adam()
        self.model.compile(optimizer= optimizer, loss='mse')
        
        action_gradients = K.gradients(Q_values, actions)
        
        self.get_action_gradients=K.function(inputs=[*self.model.input, K.learning_phase()],outputs=action_gradients)