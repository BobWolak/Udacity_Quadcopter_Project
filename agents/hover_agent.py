import math
import numpy as np
import os
import pandas as pd
import h5py
from quad_controller_rl import util

from .critic_network_hover import Critic_Hover
from .actor_network_hover import Actor_Hover
from .replay_memory_hover import Replay_Buffer_Hover
from .noise_generator import OUNoise
from quad_controller_rl.agents.base_agent import BaseAgent


class Hover_DDPG(BaseAgent):
    def __init__(self,task):
        self.task=task
        self.state_size=3
        self.state_range=self.task.observation_space.high[0:3]-self.task.observation_space.low[0:3]
        self.action_size=3
        #self.action_size=np.prod(self.task.action_space.shape)
        self.action_range=self.task.action_space.high[0:3]-self.task.action_space.low[0:3]
        
       
        
       
        self.load_weights=True
        self.save_weights_every=10
        self.model_dir=util.get_param('out')
        self.model_name="Hover_DDPG"
        self.model_ext=".h5"
        if self.load_weights or self.save_weights_every:
            self.actor_filename=os.path.join(self.model_dir,
                                             "{}_actor{}".format(self.model_name,
                                                                 self.model_ext))
            self.critic_filename=os.path.join(self.model_dir,                            "{}_critic{}".format(self.model_name,self.model_ext))
            print("Actor filename:", self.actor_filename)
            print("Critic filename:", self.critic_filename)
            
            
         
        #Save episode stats
        self.stats_filename=os.path.join(
            util.get_param('out'),"stats_{}.csv".format(self.model_name))
        self.stats_columns=['episode', 'total_reward']
        self.episode_num=1
        print("Saving stats {} to {}".format
              (self.stats_columns,self.stats_filename))    
            
        #Random Noise Generator
        self.noise=OUNoise(self.action_size)
       
            
        #Actor (Policy) Model
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        self.actor_local = Actor_Hover(self.state_size,self.action_size,
                                             self.action_low,self.action_high)
        self.actor_target = Actor_Hover(self.state_size,self.action_size,
                                             self.action_low,self.action_high)
   
        #Critic (Value) Model
        self.critic_local = Critic_Hover(self.state_size, self.action_size)
        self.critic_target = Critic_Hover(self.state_size, self.action_size)
            
        #Load pre-trained model weights
        if self.load_weights and os.path.isfile(self.actor_filename):
            try:
                self.actor_local.model.load_weights(self.actor_filename)
                self.critic_local.model.load_weights(self.critic_filename)
                print("Model weights loaded from file!")
            except Exception as e:
                print("Unable to load model weights from file!")
                print("{}: {}".format(e.__class__.__name__, str(e)))
                    
        if self.save_weights_every:
            print("Saving model weights", "every {} episodes".format(
                    self.save_weights_every) if self.save_weights_every else 
                      "disabled")
            
        #Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
            
          
        #Episode Variables
        self.episode = 0
        self.reset_episode_vars()
          
        
    def reset_episode_vars(self):
        self.episode += 1
        self.last_state = None
        self.last_action = None
        self.total_reward=0.0
                 
            
        #Replay Memory
        self.buffer_size = 100000
        self.batch_size= 64
        self.memory = Replay_Buffer_Hover(self.buffer_size)
             
        #Program Paramenters
        self.gamma= 0.99
        self.tau =0.001
             
        #Save episode stats
        #self.stats_filename=os.path.join(
         #   util.get_param('out'),"stats_{}.csv".format(self.model_name))
        #self.stats_columns=['episode', 'total_reward']
        #self.episode_num=1
        #print("Saving stats {} to {}".format
         #     (self.stats_columns,self.stats_filename))
        
             
    def write_stats(self,stats):
        df_stats=pd.DataFrame([stats],columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a',index=False,
                        header=not os.path.isfile(self.stats_filename))
            
    def preprocess_state(self,state):
        return state[0:3]
    
    def postprocess_action(self,action):
        complete_action=np.zeros(self.task.action_space.shape)
        complete_action[0:3]=action
        return complete_action
        
        
        
    def step(self, state, reward, done):
        #Transform state vector
        state = self.preprocess_state(state)
        state = state.reshape(1, -1)
             
        #Choose an Action
        action = self.act(state)
             
        #Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state,done)
            self.total_reward += reward
           
                 
                
        #Learn if samples availible
        if len(self.memory)> self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
                     
        
        if done:
            if self.save_weights_every and self.episode % self.save_weights_every==0:
                self.actor_local.model.save_weights(self.actor_filename)
                self.critic_local.model.save_weights(self.critic_filename)
                print("Model weights saved at episode", self.episode)
            # Write Stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            print("Episode #: {}, Reward:{}" .format
                  (self.episode_num, self.total_reward))
            self.reset_episode_vars()
         
        self.last_state=state
        self.last_action=action
        return self.postprocess_action(action)
        
    def act(self, states):
        states = np.reshape(states,[-1,self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample()
         
    def learn(self, experiences):
            
        #Convert experience tuples to separate arrays for each element
        states=np.vstack([e.state for e in experiences if e is not None])
        actions=np.array([e.action for e in experiences if e is not
                          None]).astype(np.float32).reshape(-1,self.action_size)
        rewards=np.array([e.reward for e in experiences if e is not
                          None]).astype(np.float32).reshape(-1,1)
        dones=np.array([e.done for e in experiences if e is not
                          None]).astype(np.uint8).reshape(-1,1)
        next_states=np.vstack([e.next_state for e in experiences if e is not None])
             
            
        #Predicted next-state actions and Q Values
        actions_next =self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next=self.critic_target.model.predict_on_batch([next_states,
                                                                      actions_next])
             
        #Compute Q Targets for current states and train local critic model
        Q_targets=rewards+self.gamma*Q_targets_next*(1-dones)
        self.critic_local.model.train_on_batch(x=[states,actions], y=Q_targets)
             
        #Train Local Actor
        action_gradients=np.reshape(self.critic_local.get_action_gradients
                                    ([states,actions,0]),(-1,self.action_size))
        self.actor_local.train_fn([states, action_gradients,1])
                                         
        #Soft-Update Target Models
        self.soft_update(self.critic_local.model,self.critic_target.model)
        self.soft_update(self.actor_local.model,self.actor_target.model)
                                         
    def soft_update(self,local_model,target_model):
        local_weights=np.array(local_model.get_weights())
        target_weights=np.array(target_model.get_weights())
                                        
        new_weights=self.tau*local_weights+(1-self.tau)*target_weights
        target_model.set_weights(new_weights)
                
                                        
                                        
            
            
                

            
                
            
                    
                    
            