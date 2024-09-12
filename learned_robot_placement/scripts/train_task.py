# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from datetime import datetime
import argparse

import math

import numpy as np
import hydra
from omegaconf import DictConfig

import pickle

import learned_robot_placement
from learned_robot_placement.utils.hydra_cfg.hydra_utils import *
from learned_robot_placement.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from learned_robot_placement.utils.task_util import initialize_task
from learned_robot_placement.envs.isaac_env_mushroom import IsaacEnvMushroom
from learned_robot_placement.models.pointnet import PointNetEncoder
from learned_robot_placement.models.pointnet2 import PointNet2Encoder
from learned_robot_placement.models.encoder.voxels import VoxelEncoder
from learned_robot_placement.models.actor_rnn import ActorNetwork_RNN
from learned_robot_placement.models.critic_rnn import CriticNetwork_RNN
from learned_robot_placement.models.actor import ActorNetwork
from learned_robot_placement.models.critic import CriticNetwork

from learned_robot_placement.models.utils.rnn_utils import RNNinternalStateCallbackStep, RNNinternalStateCallbackFit


# Use Mushroom RL library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import *
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from tqdm import trange

from experiment_launcher import run_experiment

# (Optional) Logging with Weights & biases
from learned_robot_placement.utils.wandb_utils import wandbLogger
import wandb


def experiment(cfg: DictConfig = None, cfg_file_path: str = "", seed: int = 0, results_dir: str = ""):
    # Get configs
    if(cfg_file_path):
        # Get config file from path
        cfg = OmegaConf.load(cfg_file_path)
    if(cfg.checkpoint):
        print("Loading task and train config from checkpoint config file....")
        try:
            cfg_new = OmegaConf.load(os.path.join(os.path.dirname(cfg.checkpoint), 'config.yaml'))
            cfg.task = cfg_new.task
            cfg.train = cfg_new.train
        except Exception as e:
            print("Loading checkpoint config failed!")
            print(e)
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    headless = cfg.headless
    render = cfg.render
    sim_app_cfg_path = cfg.sim_app_cfg_path
    rl_params_cfg = cfg.train.params.config
    algo_map = {"SAC_hybrid":SAC_hybrid,    # Mappings from strings to algorithms
                "BHyRL":BHyRL,}
    algo = algo_map[cfg.train.params.algo.name]

    # Set up environment
    env = IsaacEnvMushroom(headless=headless,render=render,sim_app_cfg_path=sim_app_cfg_path)
    task = initialize_task(cfg_dict, env)
    exp_name = cfg.train.params.config.name
    exp_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # append datetime for logging
    # # Set up logging paths/directories
    if not cfg.results_dir:
        results_dir = os.path.join(learned_robot_placement.__path__[0],'logs',cfg.task.name,exp_name)
    else: 
        results_dir = cfg.results_dir
    
    if(cfg.test): results_dir = os.path.join(results_dir,'test')
    if not cfg.results_dir:
        results_dir = os.path.join(results_dir,exp_stamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # log experiment config
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Test/Train use this to test/visualize current behaviour
    if(cfg.test):
        if(cfg.checkpoint):
            np.random.seed()
            # Logger
            logger = Logger(results_dir=results_dir, log_console=True)
            logger.strong_line()
            logger.info(f'Test: {exp_name}')
            logger.info(f'Test: Agent stored at '+ cfg.checkpoint)
        
            # Algorithm
            agent = algo.load(cfg.checkpoint)
            # Runner
            core = Core(agent, env)

            env._run_sim_rendering = ((not cfg.headless) or cfg.render)
            dataset = core.evaluate(n_episodes=50, render=cfg.render)

            J = np.mean(compute_J(dataset, env.info.gamma))
            R = np.mean(compute_J(dataset))
            s, *_ = parse_dataset(dataset)
            s = np.float32(s)
            E = 0
            E = agent.policy.entropy(s)
            logger.info("Test: J="+str(J)+", R="+str(R)+", E="+str(E))
        else:
            raise TypeError("Test option chosen but no valid checkpoint provided")
        env._simulation_app.close()
    else:
        if cfg.task.env.obs_representation == "stack":
            channels = cfg.task.obs.visual.stack.channels
        elif cfg.task.env.obs_representation == "tsdf":
            channels = 4
        else:
            channels = 3

        use_cuda = ('cuda' in cfg.rl_device)
        
        # Approximators
        if cfg.task.obs.visual.encoder == 'pointnet':
            embedding_actor = PointNetEncoder(channel=channels)
            embedding_critic = PointNetEncoder(channel=channels)
        elif cfg.task.obs.visual.encoder == 'pointnet2':
            embedding_actor = PointNet2Encoder(channels, False)
            embedding_critic = PointNet2Encoder(channels, False)
        else:
            embedding_actor = VoxelEncoder()
            embedding_critic = VoxelEncoder()

        
        # Need to set these for hybrid action space!
        action_space_continous = (cfg.task.env.continous_actions,)
        action_space_discrete = (cfg.task.env.discrete_actions,)
        num_actions = action_space_continous[0] + action_space_discrete[0]
        rl_state_shape = task.robot_state_len
        actor_input_shape = (256+rl_state_shape,) #pointnet embedding + 3 goal pos + 4 goal quat
        # Discrete approximator takes state and continuous action as input
        actor_discrete_input_shape = (actor_input_shape[0]+action_space_continous[0],)
        actor_mu_params = dict(network=ActorNetwork,
                            n_features=rl_params_cfg.n_features,
                            input_shape=actor_input_shape,
                            output_shape=action_space_continous,
                            use_cuda=use_cuda,
                            embedding=embedding_actor,
                            map_size=cfg.task.obs.robot_state_history_len,
                            channels=channels,
                            rl_state_shape=rl_state_shape)
        actor_sigma_params = dict(network=ActorNetwork,
                                n_features=rl_params_cfg.n_features,
                                input_shape=actor_input_shape,
                                output_shape=action_space_continous,
                                use_cuda=use_cuda,
                                embedding=embedding_actor,
                                map_size=cfg.task.obs.robot_state_history_len,
                                channels=channels,
                                rl_state_shape=rl_state_shape)
        actor_discrete_params = dict(network=ActorNetwork,
                                n_features=rl_params_cfg.n_features,
                                input_shape=actor_discrete_input_shape,
                                output_shape=action_space_discrete,
                                use_cuda=use_cuda,
                                embedding=embedding_actor,
                                map_size=cfg.task.obs.robot_state_history_len,
                                channels=channels,
                                rl_state_shape=rl_state_shape + action_space_continous[0])
        actor_optimizer = {'class': optim.Adam,
                        'params': {'lr': rl_params_cfg.lr_actor_net}}

        critic_input_shape = (actor_input_shape[0] + env.info.action_space.shape[0],) # full action space
        critic_params = dict(network=CriticNetwork,
                            optimizer={'class': optim.Adam,
                                        'params': {'lr': rl_params_cfg.lr_critic_net}},
                            loss=F.mse_loss,
                            n_features=rl_params_cfg.n_features,
                            input_shape=critic_input_shape,
                            output_shape=(1,),
                            use_cuda=use_cuda,
                            embedding=embedding_critic,
                            map_size=cfg.task.obs.robot_state_history_len,
                            channels=channels,
                            rl_state_shape=rl_state_shape,
                            )

                
        if cfg.slurm.chain_jobs.chained:
            seed_start_idx = cfg.slurm.chain_jobs.resume_from_seed
            
        # Loop over num_seq_seeds:
        for exp in range(seed_start_idx, cfg.num_seeds):

            if cfg.slurm.chain_jobs.random_seed_set:
                with open(os.path.join(results_dir, "seed.pkl"), "rb") as handle:
                     seed = pickle.load(handle)
                     np.random.set_state(seed)
            else:
                np.random.seed()
                with open(os.path.join(results_dir, "seed.pkl"), 'wb') as handle:
                    pickle.dump(np.random.get_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                cfg.slurm.chain_jobs.random_seed_set = True
                with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
                    f.write(OmegaConf.to_yaml(cfg))

            # Logger
            log_name = results_dir
            logger = Logger(log_name=log_name, results_dir=results_dir, log_console=True)
            logger.strong_line()
            logger.info(f'Experiment: {exp_name}, Trial: {exp}')
            exp_eval_dataset = list() # This will be a list of dicts with datasets from every epoch
            if cfg.wandb_activate:
                exp_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') # append datetime for logging
                wandb_logger = wandbLogger(exp_config=cfg, run_name=cfg.experiment + "_" + exp_stamp, group_name=cfg.experiment) # Optional
            
            if cfg.checkpoint:
                agent = algo.load(cfg.checkpoint)
            else:
                # Agent
                agent = algo(env.info, actor_mu_params, actor_sigma_params, actor_discrete_params, actor_optimizer, critic_params,
                            batch_size=rl_params_cfg.batch_size, initial_replay_size=rl_params_cfg.initial_replay_size,
                            max_replay_size=rl_params_cfg.max_replay_size, warmup_transitions=rl_params_cfg.warmup_transitions,
                            tau=rl_params_cfg.tau, lr_alpha=rl_params_cfg.lr_alpha, temperature=rl_params_cfg.temperature, log_std_min=rl_params_cfg.log_std_min,)
                            #gauss_noise_cov=rl_params_cfg.gauss_noise_cov)
                # Setup boosting (for BHyRL):
                # if rl_params_cfg.prior_agents is not None:
                #     prior_agents = list()
                #     for agent_path in rl_params_cfg.prior_agents:
                #         prior_agents.append(algo.load(os.path.join(learned_robot_placement.__path__[0],agent_path)))
                #    agent.setup_boosting(prior_agents='', use_kl_on_pi=rl_params_cfg.use_kl_on_pi, kl_on_pi_alpha=rl_params_cfg.kl_on_pi_alpha)
            
            # Algorithm
            core = Core(agent, env)

            # RUN
            eval_dataset = core.evaluate(n_steps=rl_params_cfg.n_steps_test, render=cfg.render, get_renders=cfg.get_renders)
            s, _, _, _, _, info, last = parse_dataset(eval_dataset)
            s = np.float32(s)
            J = np.mean(compute_J(eval_dataset, env.info.gamma))
            R = np.mean(compute_J(eval_dataset))
            E = agent.policy.entropy(s)
            collisions = task._n_collisions
            success_rate = np.sum(info)/np.sum(last) # info contains successes. rate=num_successes/num_episodes
            avg_episode_length = rl_params_cfg.n_steps_test/np.sum(last)
            logger.epoch_info(0, success_rate=success_rate, J=J, R=R, entropy=E, avg_episode_length=avg_episode_length, collisions=collisions)
            # Optional wandb logging
            #wandb.watch(agent.policy._mu_approximator.model.network)
            #wandb.watch(agent.policy._sigma_approximator.model.network)
            #wandb.watch(agent._critic_approximator.model._model[0].network)
            exp_eval_dataset.append({"Epoch": 0, "success_rate": success_rate, "J": J, "R": R, "entropy": E, "avg_episode_length": avg_episode_length, "collisions": collisions})
            
            # initialize replay buffer
            replay_buffer_init=False
            epochs = rl_params_cfg.n_epochs
            if cfg.slurm.chain_jobs.chained:
                epoch_start = cfg.slurm.chain_jobs.resume_from_epoch
                # if not first epoch, load replay buffer from last job
                if epoch_start > 0:
                    filename = 'replay_memory.zip'
                    path = os.path.join(results_dir, filename)
                    agent._replay_memory = agent._replay_memory.load(path)
                    replay_buffer_init = True
            if not replay_buffer_init:
                core.learn(n_steps=rl_params_cfg.initial_replay_size, n_steps_per_fit=rl_params_cfg.initial_replay_size, render=cfg.render)
                
            for n in trange(epoch_start, epochs, leave=False):
                # core.agent.policy._temperature *= decay 
                core.learn(n_steps=rl_params_cfg.n_steps, n_steps_per_fit=1, render=cfg.render, get_renders=cfg.get_renders)
                task._n_collisions = 0
                eval_dataset = core.evaluate(n_steps=rl_params_cfg.n_steps_test, render=cfg.render, get_renders=cfg.get_renders)
                #task.print_stats()
                s, _, _, _, _, info, last = parse_dataset(eval_dataset)
                if s.dtype == "float64": # TODO add dtype as param to method so no casting necessary
                    s = np.float32(s)
                J = np.mean(compute_J(eval_dataset, env.info.gamma))
                R = np.mean(compute_J(eval_dataset))
                E = agent.policy.entropy(s)
                success_rate = np.sum(info)/np.sum(last) # info contains successes. rate=num_successes/num_episodes
                avg_episode_length = rl_params_cfg.n_steps_test/np.sum(last)
                q_loss = core.agent._critic_approximator[0].loss_fit
                actor_loss = core.agent._actor_last_loss
                collisions = task._n_collisions

                logger.epoch_info(n+1, success_rate=success_rate, J=J, R=R, entropy=E, avg_episode_length=avg_episode_length,
                                q_loss=q_loss, actor_loss=actor_loss, collisions=collisions)
                if(rl_params_cfg.log_checkpoints):
                    logger.log_agent(agent, epoch=-1) # Log agent every epoch
                    logger.log_best_agent(agent, J) # Log best agent
                current_log={"success_rate": success_rate, "J": J, "R": R, "entropy": E, 
                            "avg_episode_length": avg_episode_length, "q_loss": q_loss, "actor_loss": actor_loss, "collisions": collisions}
                exp_eval_dataset.append(current_log)
                if cfg.wandb_activate:
                    wandb_logger.run_log_wandb(success_rate, J, R, E, avg_episode_length, q_loss, collisions)

                if cfg.slurm.chain_jobs.chained:
                    # save replay memory for chained jobs
                    filename = 'replay_memory.zip'
                    path = os.path.join(results_dir, filename)
                    agent._replay_memory.save(path, full_save=True)
                    # save current epoch for chained jobs
                    cfg.slurm.chain_jobs.resume_from_epoch =  n + 1
                    cfg.checkpoint = os.path.join(results_dir, 'agent--1.msh')
                    # log experiment config
                    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
                        f.write(OmegaConf.to_yaml(cfg))
                
            # Get video snippet of final learnt behavior (enable internal rendering for this)
            #prev_env_render_setting = bool(env._run_sim_rendering)
            #env._run_sim_rendering = True
            ##img_dataset = core.evaluate(n_episodes=5, get_renders=True)
            #env._run_sim_rendering = prev_env_render_setting
            # log dataset and video
            if cfg.slurm.chain_jobs.chained:
                cfg.slurm.chain_jobs.resume_from_seed = exp + 1
                cfg.slurm.chain_jobs.resume_from_epoch =  0
                cfg.checkpoint = ''
                cfg.slurm.chain_jobs.random_seed_set = False
                # log experiment config
                with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
                    f.write(OmegaConf.to_yaml(cfg))
            logger.log_dataset(exp_eval_dataset)
            # wandb_logger.run_log_wandb(exp_config=cfg, run_name=logger._log_id, group_name=exp_name+'_'+exp_stamp, dataset=exp_eval_dataset)
            ##img_dataset = img_dataset[::15] # Reduce size of img_dataset. Take every 15th image
            #if cfg.wandb_activate:
                #wandb_logger.vid_log_wandb(img_dataset=img_dataset)

    # Shutdown
    env._simulation_app.close()


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_run_exp(cfg: DictConfig):
    experiment(cfg)


if __name__ == '__main__':
    #parse_hydra_configs_and_run_exp()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('results_dir')
    #parser.add_argument('seed')
    #parser.add_argument('cfg_file_path')
    #args = parser.parse_args()
    #experiment(cfg_file_path = args.cfg_file_path, seed=args.seed, results_dir=args.results_dir)
    parser = argparse.ArgumentParser("Train/Test agent: (Local or SLURM)")
    #parser.add_argument("--cfg_file_path", type=str, default="/home/sabin/models_rlmmbp/config.yaml", help="Optional config file to run experiment (typically when using slurm)")
    parser.add_argument("--cfg_file_path", type=str, default="", help="Optional config file to run experiment (typically when using slurm)")
    args, _ = parser.parse_known_args()
    
    if args.cfg_file_path:
        # Leave below unchanged for slurm 'experiment_launcher'
        # (https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher/)
        parser.add_argument('--seed', type=int)
        parser.add_argument('--results_dir', type=str)
        args = parser.parse_args()
        
        run_experiment(experiment, vars(args))
    else:
        parse_hydra_configs_and_run_exp()


# if __name__ == '__main__':
#     #parse_hydra_configs_and_run_exp()
#     #parser = argparse.ArgumentParser()
#     #parser.add_argument('results_dir')
#     #parser.add_argument('seed')
#     #parser.add_argument('cfg_file_path')
#     #args = parser.parse_args()
#     #experiment(cfg_file_path = args.cfg_file_path, seed=args.seed, results_dir=args.results_dir)
#     parser = argparse.ArgumentParser("Train/Test agent: (Local or SLURM)")
#     parser.add_argument("--cfg_file_path", type=str, default="/home/sabin/IAS/rlmmbp/config.yaml", help="Optional config file to run experiment (typically when using slurm)")
#     args, _ = parser.parse_known_args()
    
#     if args.cfg_file_path is not None:
#         experiment(cfg_file_path = args.cfg_file_path)

#     else:
#         parse_hydra_configs_and_run_exp()

