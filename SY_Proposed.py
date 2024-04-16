# %% [markdown]
# SY TX 3-6 [2,3,4,5]
# XH TX 1 2 7 8 [0,1,6,7] 
# 

# %% [markdown]
# DQN - XH 

# %%
# %%
# import libraries
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from torch.utils.tensorboard import SummaryWriter
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from typing import Callable
import time, math
import random
from env_noise import Rayleigh_Noise, System_bias, Device_shadowing, Rayleigh_Noise_single

# Configurations
FILEPATH = 'sy2.4G_c.npy'
FILEPATH_origin = 'sy2.4G_s.npy'
Tx_ID = np.array([1,2,3,4])
# Noise parameters
Bias = 3
Rayleigh_factor = 3

error_mat = np.zeros((39,15,10))

# %%
class Env_3D_train(gym.Env):
    """
    Custom Environment that follows gym interface. The 3D!! localization environment for the agent to learn.
    Replicating paper "Deep Reinforcement Learning (DRL): Another Perspective for Unsupervised Wireless Localization"
    https://ieeexplore.ieee.org/document/8924617/citations?tabFilter=papers#citations
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, 
                 RSSIfilepath = FILEPATH,
                 render_mode="console",
                 Relocate = False):
        
        super(Env_3D_train, self).__init__()
        self.render_mode = render_mode
        self.Relocate = Relocate

        # Load data
        DATA = np.load(RSSIfilepath)
        self.Tx_id = Tx_ID
        self.scenario_id = 0 # 0 represents Siyuan, 1 reprensents Xuehuo
        self.data = DATA[self.Tx_id,:,:,:]; self.data = np.around(self.data)
        DATA_o = np.load(FILEPATH_origin); self.data_original = DATA_o[self.Tx_id,:,:,:];self.data_original = np.around(self.data_original)

        self.size= np.array(self.data.shape[1:])
        self.AP_num = self.data.shape[0]
        
        # The observation contains the previous location of agent (Previous_Location) and received RSSI (Next_RSSI) from APs at the next coordinate
        self.observation_space = spaces.Dict(   
                    {

                        'Previous_Location': spaces.Box(low=np.array([[0,0,0]]), 
                                                        high=self.size[np.newaxis,:], 
                                                        shape=(1,3), 
                                                        dtype=int),
                        'Next_RSSI': spaces.Box(low=-200, high=200, 
                                                    shape=(1,self.AP_num), 
                                                    dtype=np.float64) 
                    }
        )

        # Define the action space
        n_actions = 7 #
        self.action_space = spaces.Discrete(n_actions)

        # The following dictionary maps abstract actions from `self.action_space` to the direction walk in if that action is taken.
        # The current coordinate of the agent is to to be decided by the policy 
        self._action_to_direction = {
            0: np.array([0, 0, 0]),    # stop
            1: np.array([1, 0, 0]),    # right
            2: np.array([-1, 0, 0]),   # left
            3: np.array([0, 1, 0]),    # front
            4: np.array([0, -1, 0]),   # back
            5: np.array([0, 0, 1]),    # up
            6: np.array([0, 0, -1]),   # down
        }

    # Default function to get observations
    def _get_obs(self):
        return {"Previous_Location": self._prev_loc, "Next_RSSI": self._next_RSSI}
    
    def _get_info(self):
        return {"Previous_Location": self._prev_loc, 
                 "Next_Location": self._next_loc,
                 "Prev_RSSI": self._prev_RSSI,
                 "Next_RSSI": self._next_RSSI,
                 "is_success": self._success_indicator
                                                   } 

    def initial_location_generator(self, arraysize):
        initial_location = np.zeros((1,3),int)
        initial_location[0,0] += np.random.randint(arraysize[0])
        initial_location[0,1] += np.random.randint(arraysize[1])
        initial_location[0,2] += np.random.randint(arraysize[2])
        return initial_location

    def next_location_generator(self,prev_loc):
        next_loc = prev_loc
        choice = random.choice([0, 1, 2])
        next_loc[0,choice] = next_loc[0,choice] + np.random.randint(-1, 2,size = 1)
        next_loc = np.clip(next_loc,[0,0,0],self.size-[1,1,1])
        return np.array(next_loc)
    
    def cube_RSSI_random(self):
        x = self._next_loc[0,0]; y = self._next_loc[0,1]; z = self._next_loc[0,2]
        original_cube = self.data_original[:,3*x:3*x+3,3*y:3*y+3,2*z:2*z+2]
        init_loc_cube = self.initial_location_generator(original_cube.shape[1:]); init_loc_cube = init_loc_cube[0,:]
        true_next_RSSI = original_cube[:,init_loc_cube[0],init_loc_cube[1],init_loc_cube[2]];true_next_RSSI = true_next_RSSI[np.newaxis,:]
        self.wrong_loc = [3*self._next_loc[0,0]+init_loc_cube[0],3*self._next_loc[0,1]+init_loc_cube[1],2*self._next_loc[0,2]+init_loc_cube[2]]
        return true_next_RSSI
    
    def direction_survey(self):
        survey_book = np.zeros((7,self.AP_num))
        survey_book[0,:] = np.zeros((self.AP_num))
        for i in [1,2,3,4,5,6]:
            fake_new_loc = self._prev_loc + self._action_to_direction[i];fake_new_loc = np.clip(fake_new_loc, [0,0,0], self.size-[1,1,1])
            survey_book[i,:] = self.data[:,fake_new_loc[0][0],fake_new_loc[0][1],fake_new_loc[0][2]] - self._prev_RSSI
        survey_book += 0.01; next_rssi = self._next_RSSI + 0.01
        best_similarity =  (survey_book[0,:]@next_rssi[0,:])/(np.linalg.norm(survey_book[0,:])*np.linalg.norm(next_rssi))
        best_index = 0
        for i in np.arange(7):
            similarity = (survey_book[i,:]@next_rssi[0,:])/(np.linalg.norm(survey_book[i,:])*np.linalg.norm(next_rssi))
            if (similarity - best_similarity) > 0.05:
                best_similarity = similarity
                best_index = i
        return best_index, survey_book[best_index,:]

    # reset function
    def reset(self, seed=None, options=None):

        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=None, options=options)
        self._success_indicator = False
        
        # Choose the agent's location uniformly at random
        LOC = self.initial_location_generator(self.size)
        self._prev_loc = LOC; self._prev_loc = np.clip(self._prev_loc, [0,0,0], self.size-[1,1,1])
        self._next_loc = LOC; self._next_loc = np.clip(self._next_loc, [0,0,0], self.size-[1,1,1])

        cdnt = self._prev_loc.reshape(3,1)
        self._prev_RSSI = self.data[:,cdnt[0],cdnt[1],cdnt[2]].transpose()

        self._next_loc = self.next_location_generator(self._next_loc)
        cdnt = self._next_loc.reshape(3,1)
        self._next_RSSI = self.data[:,cdnt[0],cdnt[1],cdnt[2]].transpose()

        # Difference - center to center
        # self._next_RSSI = self._next_RSSI - self._prev_RSSI

        # If no shadowing - point to center
        # self._next_RSSI = self.cube_RSSI_random() 
        # self._next_RSSI = self._next_RSSI - self._prev_RSSI
        # best_index, diff = self.direction_survey(); self._next_RSSI = self._next_RSSI*0.7 + diff*0.3

        # If Device shadowing, must with expanded dimension on
        self._next_RSSI = self.cube_RSSI_random()- self._prev_RSSI; 
        # p_loc = np.array([0,0]); p_loc[0]=self._prev_loc[0,0]*3; p_loc[1]=self._prev_loc[0,1]*3
        # n_loc = np.array([0,0]); n_loc[0]=self._next_loc[0,0]*3; n_loc[1]=self._next_loc[0,1]*3
        # self._prev_RSSI, self._next_RSSI = Device_shadowing(
        #     self.scenario_id, self.Tx_id, p_loc, n_loc, self._prev_RSSI[0,:], self._next_RSSI[0,:])
        # self._next_RSSI = np.expand_dims(self._next_RSSI,axis=0);  self._prev_RSSI = np.expand_dims(self._prev_RSSI,axis=0)
        # best_index, diff = self.direction_survey(); self._next_RSSI = self._next_RSSI*0.7 + diff*0.3

        # # Rayleigh noise
        dim = self._prev_RSSI.shape[0]
        # a = Rayleigh_Noise_single(dim); self._prev_RSSI += -a
        a = Rayleigh_Noise_single(dim); b = Rayleigh_Noise_single(dim); self._next_RSSI += -(a-b)



        # Return the observation
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    # Step function
    def step(self, action):
        
        # Map the action (element 0-9) to the direction to walk in
        move = self._action_to_direction[action]

        # To make sure the agent don't leave the grid
        recorder = np.clip(self._prev_loc+move,[0,0,0],self.size-[1,1,1])
        
        # Evaluate whether the agent have reached the goal
        terminated = bool((recorder == self._next_loc).all())
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if terminated else -0.1*(abs(np.linalg.norm(recorder-self._next_loc)))
        self._success_indicator = True if terminated else False
        # error_mat[self.wrong_loc[0],self.wrong_loc[1],self.wrong_loc[2]] += 0.00001

        # Retrieve the new observaion
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        pass

# %%
# env = Env_3D_train()
# a = 0; test_number = 1000000
# for i in np.arange(test_number):
#     obs, info = env.reset()
#     best_index,xx = env.direction_survey()
#     if np.sum(info['Next_Location'] - info['Previous_Location'] + env._action_to_direction[best_index]) == 0:
#         a = a + 1
# print(a/test_number)

# %%
register(id="img_v1", entry_point=Env_3D_train, max_episode_steps= 50)
env = gym.make('img_v1')
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

# %%
class Env_3D_test(gym.Env):
    """
    Custom Environment that follows gym interface. The 3D!! localization environment for the agent to learn.
    Replicating paper "Deep Reinforcement Learning (DRL): Another Perspective for Unsupervised Wireless Localization"
    https://ieeexplore.ieee.org/document/8924617/citations?tabFilter=papers#citations
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, 
                 RSSIfilepath = FILEPATH,
                 render_mode="console",
                 Relocate = False):
        
        super(Env_3D_test, self).__init__()
        self.render_mode = render_mode
        self.Relocate = Relocate

        # Load data
        DATA = np.load(RSSIfilepath)
        self.Tx_id = Tx_ID
        self.scenario_id = 0 # 0 represents Siyuan, 1 reprensents Xuehuo
        self.data = DATA[self.Tx_id,:,:,:]; self.data = np.around(self.data)
        DATA_o = np.load(FILEPATH_origin); self.data_original = DATA_o[self.Tx_id,:,:,:];self.data_original = np.around(self.data_original)

        self.size= np.array(self.data.shape[1:])
        self.AP_num = self.data.shape[0]
        
        # The observation contains the previous location of agent (Previous_Location) and received RSSI (Next_RSSI) from APs at the next coordinate
        self.observation_space = spaces.Dict(   
                    {

                        'Previous_Location': spaces.Box(low=np.array([[0,0,0]]), 
                                                        high=self.size[np.newaxis,:], 
                                                        shape=(1,3), 
                                                        dtype=int),
                        'Next_RSSI': spaces.Box(low=-200, high=200, 
                                                    shape=(1,self.AP_num), 
                                                    dtype=np.float64) 
                    }
        )

        # Define the action space
        n_actions = 7 #
        self.action_space = spaces.Discrete(n_actions)

        # The following dictionary maps abstract actions from `self.action_space` to the direction walk in if that action is taken.
        # The current coordinate of the agent is to to be decided by the policy 
        self._action_to_direction = {
            0: np.array([0, 0, 0]),    # stop
            1: np.array([1, 0, 0]),    # right
            2: np.array([-1, 0, 0]),   # left
            3: np.array([0, 1, 0]),    # front
            4: np.array([0, -1, 0]),   # back
            5: np.array([0, 0, 1]),    # up
            6: np.array([0, 0, -1]),   # down
        }

    # Default function to get observations
    def _get_obs(self):
        return {"Previous_Location": self._prev_loc, "Next_RSSI": self._next_RSSI}
    
    def _get_info(self):
        return {"Previous_Location": self._prev_loc, 
                 "Next_Location": self._next_loc,
                 "Prev_RSSI": self._prev_RSSI,
                 "Next_RSSI": self._next_RSSI,
                 "is_success": self._success_indicator
                                                   } 

    def initial_location_generator(self, arraysize):
        initial_location = np.zeros((1,3),int)
        initial_location[0,0] += np.random.randint(arraysize[0])
        initial_location[0,1] += np.random.randint(arraysize[1])
        initial_location[0,2] += np.random.randint(arraysize[2])
        return initial_location

    def next_location_generator(self,prev_loc):
        next_loc = prev_loc
        choice = random.choice([0, 1, 2])
        next_loc[0,choice] = next_loc[0,choice] + np.random.randint(-1, 2,size = 1)
        next_loc = np.clip(next_loc,[0,0,0],self.size-[1,1,1])
        return np.array(next_loc)
    
    def cube_RSSI_random(self):
        x = self._next_loc[0,0]; y = self._next_loc[0,1]; z = self._next_loc[0,2]
        original_cube = self.data_original[:,3*x:3*x+3,3*y:3*y+3,2*z:2*z+2]
        init_loc_cube = self.initial_location_generator(original_cube.shape[1:]); init_loc_cube = init_loc_cube[0,:]
        true_next_RSSI = original_cube[:,init_loc_cube[0],init_loc_cube[1],init_loc_cube[2]];true_next_RSSI = true_next_RSSI[np.newaxis,:]
        self.wrong_loc = [3*self._next_loc[0,0]+init_loc_cube[0],3*self._next_loc[0,1]+init_loc_cube[1],2*self._next_loc[0,2]+init_loc_cube[2]]
        return true_next_RSSI
    
    def direction_survey(self):
        survey_book = np.zeros((7,self.AP_num))
        survey_book[0,:] = np.zeros((self.AP_num))
        for i in [1,2,3,4,5,6]:
            fake_new_loc = self._prev_loc + self._action_to_direction[i];fake_new_loc = np.clip(fake_new_loc, [0,0,0], self.size-[1,1,1])
            survey_book[i,:] = self.data[:,fake_new_loc[0][0],fake_new_loc[0][1],fake_new_loc[0][2]] - self._prev_RSSI
        survey_book += 0.01; next_rssi = self._next_RSSI + 0.01
        best_similarity =  (survey_book[0,:]@next_rssi[0,:])/(np.linalg.norm(survey_book[0,:])*np.linalg.norm(next_rssi))
        best_index = 0
        for i in np.arange(7):
            similarity = (survey_book[i,:]@next_rssi[0,:])/(np.linalg.norm(survey_book[i,:])*np.linalg.norm(next_rssi))
            if (similarity - best_similarity) > 0.05:
                best_similarity = similarity
                best_index = i
        return best_index, survey_book[best_index,:]

    # reset function
    def reset(self, seed=None, options=None):

        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=None, options=options)
        self._success_indicator = False
        
        # Choose the agent's location uniformly at random
        LOC = self.initial_location_generator(self.size)
        self._prev_loc = LOC; self._prev_loc = np.clip(self._prev_loc, [0,0,0], self.size-[1,1,1])
        self._next_loc = LOC; self._next_loc = np.clip(self._next_loc, [0,0,0], self.size-[1,1,1])

        cdnt = self._prev_loc.reshape(3,1)
        self._prev_RSSI = self.data[:,cdnt[0],cdnt[1],cdnt[2]].transpose()

        self._next_loc = self.next_location_generator(self._next_loc)
        cdnt = self._next_loc.reshape(3,1)
        self._next_RSSI = self.data[:,cdnt[0],cdnt[1],cdnt[2]].transpose()

        # Difference - center to center
        # self._next_RSSI = self._next_RSSI - self._prev_RSSI

        # If no shadowing - point to center
        # self._next_RSSI = self.cube_RSSI_random() 
        # self._next_RSSI = self._next_RSSI - self._prev_RSSI
        # best_index, diff = self.direction_survey(); self._next_RSSI = self._next_RSSI*0.7 + diff*0.3

        # If Device shadowing, must with expanded dimension on
        self._next_RSSI = self.cube_RSSI_random()- self._prev_RSSI; 
        # p_loc = np.array([0,0]); p_loc[0]=self._prev_loc[0,0]*3; p_loc[1]=self._prev_loc[0,1]*3
        # n_loc = np.array([0,0]); n_loc[0]=self._next_loc[0,0]*3; n_loc[1]=self._next_loc[0,1]*3
        # self._prev_RSSI, self._next_RSSI = Device_shadowing(
        #     self.scenario_id, self.Tx_id, p_loc, n_loc, self._prev_RSSI[0,:], self._next_RSSI[0,:])
        # self._next_RSSI = np.expand_dims(self._next_RSSI,axis=0);  self._prev_RSSI = np.expand_dims(self._prev_RSSI,axis=0)
        # best_index, diff = self.direction_survey(); self._next_RSSI = self._next_RSSI*0.7 + diff*0.3

        # # Rayleigh noise
        dim = self._prev_RSSI.shape[0]
        # a = Rayleigh_Noise_single(dim); self._prev_RSSI += -a
        a = Rayleigh_Noise_single(dim); b = Rayleigh_Noise_single(dim); self._next_RSSI += -(a-b)


        # Return the observation
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    # Step function
    def step(self, action):
        
        # Map the action (element 0-9) to the direction to walk in
        move = self._action_to_direction[action]

        # To make sure the agent don't leave the grid
        recorder = np.clip(self._prev_loc+move,[0,0,0],self.size-[1,1,1])
        
        # Evaluate whether the agent have reached the goal
        terminated = bool((recorder == self._next_loc).all())
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if terminated else -0.1*(abs(np.linalg.norm(recorder-self._next_loc)))
        self._success_indicator = True if terminated else False
        # error_mat[self.wrong_loc[0],self.wrong_loc[1],self.wrong_loc[2]] += 0.00001

        # Retrieve the new observaion
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        pass

# %%
def register_env():
    register(id="Loc-v1", entry_point=Env_3D_train, max_episode_steps= 50)
    register(id="Loc-v1-env", entry_point=Env_3D_test, max_episode_steps=1)
    return 

def train():
    register_env()
    vec_env = make_vec_env('Loc-v1', n_envs = 50) 
    eval_env = gym.make('Loc-v1-env')    
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./data/SYProposedModels/",
                                log_path="./data/BestModels/", n_eval_episodes = 50, eval_freq = 10,
                                deterministic=True, render=False)

    model = A2C("MultiInputPolicy_new", vec_env, learning_rate=0.002, n_steps = 10,device="cpu", verbose = 1, tensorboard_log="./data/log/SY_Proposed", policy_kwargs=dict(normalize_images=False)) 
    # model = DQN("MultiInputPolicy", vec_env, learning_rate=0.002,device="cuda", verbose = 1, tensorboard_log="./data/log/") 

    model.learn(total_timesteps=5000000, callback=eval_callback, progress_bar=False)

# %%
# register_env()
# model_Path = 'data/BestModels/best_model.zip'
# eval_env = gym.make('Loc-v1-env') 
# model = A2C.load(model_Path)
# obs,info = eval_env.reset()
# A2DD = {
#             0: np.array([0, 0, 0]),    # stop
#             1: np.array([1, 0, 0]),    # right
#             2: np.array([-1, 0, 0]),   # left
#             3: np.array([0, 1, 0]),    # front
#             4: np.array([0, -1, 0]),   # back
#             5: np.array([0, 0, 1]),    # up
#             6: np.array([0, 0, -1]),   # down
#         }
# A2DD[int(action)]
# action, _states = model.predict(obs)
# move = A2DD[int(action)]
# recorder = np.clip(info['Previous_Location']+move,[0,0,0],eval_env.size-[1,1,1])
# terminated = bool((recorder == info['Next_Location']).all())

# obs, rewards, dones, info = eval_env.step(action)

# %%
if __name__=='__main__':
    train()



