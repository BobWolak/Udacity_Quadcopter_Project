import numpy as np
import math
from keras import Sequential
from keras import layers, models, optimizers
from keras import backend as K

class Actor_Takeoff:
    def __init__(self,state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low  = action_low
        self.action_high = action_high
        self.action_range = self.action_high-self.action_low
        self.learning_rate= 0.001
        self.build_model()
        
        
    def build_model(self):
        states =layers.Input(shape=(self.state_size,), name='states')
        h1_net =layers.Dense(units=32)(states)
        h1_net =layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001)(h1_net)
        h1_net =layers.Activation('relu')(h1_net)
    
        h2_net =layers.Dense(units=64)(h1_net)
        h2_net =layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001)(h2_net)
        h2_net =layers.Activation('relu')(h2_net)
    
        h3_net =layers.Dense(units=32)(h2_net)
        h3_net =layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001)(h3_net)
        h3_net =layers.Activation('relu')(h3_net)
    
        raw_actions=layers.Dense(units=self.action_size,activation='sigmoid',
                                 name='raw_actions')(h3_net)

        actions=layers.Lambda(lambda x: (x*self.action_range)+self.action_low,
                              name='actions')(raw_actions)
    
        self.model =models.Model(inputs=states, outputs=actions)
        action_gradients=layers.Input(shape=(self.action_size,))
        loss= K.mean(-action_gradients*actions)
        
        optimizer= optimizers.Adam()
        updates_op= optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
    
        self.train_fn=K.function(inputs=[self.model.input,action_gradients,
                                         K.learning_phase()],outputs=
                                 [],updates=updates_op)
    