import learned_robot_placement

import os
from itertools import product

import math

import hydra
from omegaconf import DictConfig
from learned_robot_placement.utils.hydra_cfg.hydra_utils import *

import argparse
import datetime
import inspect
import os
import traceback
from copy import copy, deepcopy
from distutils.util import strtobool
from importlib import import_module

import numpy as np
from joblib import Parallel, delayed


LOCAL = False#is_local()
TEST = False

N_SEEDS = 1

N_CORES = 2

PARTITION = 'a40'
GRES = 'gpu:a40'
CONDA_ENV = 'isaac-sim'


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs_and_launch_exp(cfg: DictConfig):

    import datetime
    time = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
    exp_name = "visual_rlmmbp"
    # base_dir = os.path.join(learned_robot_placement.__path__[0],'logs', time)
    base_dir = os.path.join(os.getenv("WORK"),'logs')

    launcher = Launcher(
        exp_name='visual_rlmmbp',
        exp_file='train_task', # local path without .py    
        base_dir=base_dir,
        n_seeds=N_SEEDS,
        n_cores=N_CORES,
        hours=24,
        minutes=0,
        seconds=0,
        partition=PARTITION,
        gres=GRES,
        conda_env=CONDA_ENV,
        timestamp=time,
        compact_dirs=False
    )
    

    base_dir = os.path.join(base_dir, exp_name)

    exp_num = 0
    lr_rates = [0.0001]
    bs = [512]
    #tsdf_res = [64]
    sizes = [1,2,4]
    samples = ["distance"]
    nav_scenes = ["simple"]
    num_actions = [3]
    hist_len = [2,5,10]

    # first batch: complex scene for vox downsample with active vision and without and all sampling techniques --> 3 days --> 6 jobs --> FISRT ITER LAUNCHED
    # second batch: simple scene for vox downsample with active vision and without and all sampling techniques --> 2 days --> 6 jobs
    # third batch: voxel downsample simple scene with history len 1,5,10 --> 2 days --> 3 jobs
    # fourth batch: complex scene stack 1,2,4 with pos encoding --> 3 days --> 3 jobs
    # fifth batch: simple scene stack 1,2,4 with pos encoding--> 2 days --> 3 jobs
    # sixth batch: simple and complex stack4 with and without pos encoding --> 4 jobs
    # 7th batch: keyframes simple scene, tsdf simple and complex scene --> 3 jobs
    # 8th batch compare reward and no reward



    for dist in hist_len:
        for num_action in num_actions:

            # cfg.experiment = 'vision:#obst' + str(num_obstacles)+ '#grasp' + str(num_grasp_obj) + ',rndm_goal:' + str(random_goal) + ",obst_scale:" + obstacle_type + ',scene:' + scene_type + 'rwrd_lool_goal' + str(reward_look_at_goal)
            # cfg.experiment = 'voxel_no_bn_small_encoder_longer_horizon_10_no_rob_emb' # + str(0.0001) + '_bs_' + str(size)

            
            cfg.experiment = 'voxel_downsample_hist_len_' + str(dist)
            cfg.task.obs.visual.keyframe_voxel.num_keyframes = dist

            cfg.task.env.continous_actions = num_action

            cfg.results_dir = os.path.join(base_dir, cfg.experiment, time)
            results_dir = os.path.join(base_dir, cfg.experiment, time)
            
            # Save experiment config in a file at the right path
            os.makedirs(results_dir)
            exp_cfg_file_path = os.path.join(results_dir, 'config.yaml')
            with open(exp_cfg_file_path, 'w') as f:
                f.write(OmegaConf.to_yaml(cfg))
            
            launcher.add_experiment(cfg_file_path__=exp_cfg_file_path)
                
            exp_num +=1

    launcher.run(LOCAL, TEST)


class ResultsDirException(Exception):
    """
    Raised when the results directory already exists
    """
    def __init__(self, exp, results_dir='./logs'):
        message = f"\n" \
                  f"When trying to add the experiment: {exp}\n" \
                  f"Results directory {results_dir} already exists.\n" \
                  f"Make sure that varying parameters have a trailing double underscore."
        super().__init__(message)


class Launcher(object):
    """
    Creates and starts jobs with Joblib or SLURM.

    """

    def __init__(self, exp_name, exp_file, n_seeds, n_cores=1,
                 hours=24, minutes=0, seconds=0,
                 project_name=None, base_dir=None,
                 conda_env=None, gres=None, partition=None,
                 begin=None, timestamp=None, compact_dirs=False):
        """
        Constructor.

        Args:
            exp_name (str): name of the experiment
            exp_file (str): name of the python module running a single experiment (relative path)
            n_seeds (int): number of seeds for each experiment configuration
            n_cores (int): number of cpu cores
            hours (int): number of hours the experiment can last (in slurm)
            minutes (int): number of minutes the experiment can last (in slurm)
            seconds (int): number of seconds the experiment can last (in slurm)
            base_dir (str): path to directory to save the results (in hhlr results are saved to /work/scratch/$USER)
            n_exps_in_parallel (int): number of experiment configurations to run in parallel.
                If running in the cluster, and the gpu is selected, then it is the number of jobs in each slurm file
                (e.g. for multiple experiments in the same gpu)
            conda_env (str): name of the conda environment to run the experiments in
            gres (str): request cluster resources. E.g. to add a GPU in the IAS cluster specify gres='gpu:rtx2080:1'
            partition (str, None): the partition to use in case of slurm execution. If None, no partition is specified.
            begin (str): start the slurm experiment at a given time (see --begin in slurm docs)
            use_timestamp (bool): add a timestamp to the experiment name
            compact_dirs (bool): If true, only the parameter value is used for the directory name.

        """
        self._exp_name = exp_name
        self._exp_file = exp_file
        self._n_seeds = n_seeds
        self._n_cores = n_cores
        self._duration = Launcher._to_duration(hours, minutes, seconds)
        self._hours = hours
        self._minutes = minutes
        self._seconds = seconds
        self._conda_env = conda_env
        self._gres = gres
        self._partition = partition
        self._begin = begin
        self._timestamp = timestamp

        self._n_chained_jobs = math.ceil(hours/24) # Cluster has restriction of 24h
        if self._n_chained_jobs > 1:
            self._hours_per_job = []
            for i in range(24,hours+1,24):
                self._hours_per_job.append(24)
            if hours % 24 > 0:
                self._hours_per_job.append(hours % 24)

        self._experiment_list = list()

        base_dir = os.path.join(os.getenv("WORK"),'logs')if base_dir is None else base_dir
        self._exp_dir_local = os.path.join(base_dir, self._exp_name, self._timestamp)

        # tracks the results directories
        self._results_dir_l = []
        # os.makedirs(self._exp_dir_local, exist_ok=True)

        self._compact_dirs = compact_dirs

    def add_experiment(self, **kwargs):
        self._experiment_list.append(deepcopy(kwargs))

    def run(self, local, test=False, sequential=False):
        self._check_experiments_results_directories()
        if local:
            if sequential:
                self._run_sequential(test)
            else:
                self._run_local(test)
        else:
            self._run_slurm(test)

        self._experiment_list = list()

    def generate_slurm(self, exp_dir_slurm_files, exp_dir_slurm_logs, command_line_list=None, next_job_in_chain: str = None, duration = None):
        partition_option = ''
        gres_option = ''
        batch_code = ''
        conda_code = ''

        # wandb_offline = ''
        # wandb_offline += 'export http_proxy=http://proxy:80 \n'
        # wandb_offline += 'export https_proxy=http://proxy:80 \n'
        # wandb_offline += 'wandb login --relogin bd9a4f50662362cc463092a5f5aa69640d961c27 \n\n'
        # wandb_offline += 'wandb sync --include-synced --include-offline --sync-all \n'



        partition_option += f'#SBATCH -p {self._partition}\n'
        gres_option += '#SBATCH --gres=' + str(self._gres) + '\n'

        batch_code += 'unset SLURM_EXPORT_ENV \n \n'
        batch_code += 'module load python \n'
        batch_code += 'export http_proxy=http://proxy:80 \n'
        batch_code += 'export https_proxy=http://proxy:80 \n'
        batch_code += 'source /home/hpc/g101ea/g101ea11/isaac/ov/pkg/isaac_sim-2022.2.0/setup_conda_env.sh \n'
        conda_code += f'conda activate {self._conda_env}\n\n'
        conda_code += 'wandb login --relogin bd9a4f50662362cc463092a5f5aa69640d961c27 \n\n'
        conda_code += 'wandb online \n\n'
        python_code = f'python {self._exp_file_path} \\'

        code = f"""\
#!/bin/bash -l

###############################################################################
# SLURM Configurations

# Optional parameters
{partition_option}{gres_option}
# Mandatory parameters
#SBATCH --job-name={self._exp_name}
#SBATCH -a 0-{self._n_seeds - 1}
#SBATCH --time={duration}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={self._n_cores}
#SBATCH -o {exp_dir_slurm_logs}/%A_%a.out
#SBATCH -e {exp_dir_slurm_logs}/%A_%a.err
#SBATCH --export=NONE

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"


# BATCH CODE
{batch_code}
"""
        if next_job_in_chain is not None:
            code += f"""\
cd {exp_dir_slurm_files}           
sbatch --dependency=afterany:$SLURM_JOB_ID {next_job_in_chain}               
"""
        code += f"""\

# Program specific arguments
{conda_code}

"""
        code += f"""\
# Program specific arguments

echo "Running scripts in parallel..."
echo "########################################################################"
            
"""
        for command_line in command_line_list:
            code += f"""\
                
{python_code}
\t\t--seed $SLURM_ARRAY_TASK_ID \\
\t\t{command_line}  &

"""
        code += f"""\
            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
"""

#         code += f"""\
# {wandb_offline}   
        
# """
        return code

    def save_slurm(self, exp_dir_slurm_files, exp_dir_slurm_logs, command_line_list=None, idx_exp: str = None, idx_job_in_chain: int = None, duration = None):
        label = f"_{idx_exp}" if idx_exp is not None else ""
        if idx_job_in_chain is not None:
            label += f"_{str(idx_job_in_chain).zfill(self._n_chained_jobs)}" if idx_job_in_chain is not None else ""
        script_name = f'slurm_{self._exp_name}{label}.sh'        
        full_path = os.path.join(exp_dir_slurm_files, script_name)
        if idx_job_in_chain is not None:    
            if int(idx_job_in_chain) < self._n_chained_jobs -1:
                next_job_label = f"_{idx_exp}" if idx_exp is not None else ""
                next_job_label += f"_{str(idx_job_in_chain+1).zfill(self._n_chained_jobs)}"
                next_job_in_chain = f'slurm_{self._exp_name}{next_job_label}.sh'
                code = self.generate_slurm(exp_dir_slurm_files, exp_dir_slurm_logs, command_line_list=command_line_list, next_job_in_chain=next_job_in_chain, duration=duration)
            else: 
                code = self.generate_slurm(exp_dir_slurm_files, exp_dir_slurm_logs, command_line_list=command_line_list, duration=duration)
        else:
            code = self.generate_slurm(exp_dir_slurm_files, exp_dir_slurm_logs, command_line_list=command_line_list, duration=self._duration)
        
        with open(full_path, "w") as file:
            file.write(code)

        return full_path

    def _run_slurm(self, test):
        # Generate and save slurm files
        slurm_files_path_l = []
        experiment_list_chunked = []
        for i in range(len(self._experiment_list)):
            experiment_list_chunked.append(self._experiment_list[i])

        for i, exp in enumerate(experiment_list_chunked):
            command_line_l = []
            exp_new_without_underscore = self.remove_last_underscores_dict(exp)
            command_line_arguments = self._convert_to_command_line(exp_new_without_underscore)
            if 'cfg_file_path' in exp_new_without_underscore:
                results_dir = exp_new_without_underscore['cfg_file_path'][:-len('/config.yaml')]
                exp_dir_slurm_files = os.path.join(results_dir, "slurm_files")
                exp_dir_slurm_logs = os.path.join(results_dir, "slurm_logs")
            else:
                results_dir = self._generate_results_dir(self._exp_dir_local, i)
                exp_dir_slurm_files = os.path.join(results_dir, "slurm_files")
                exp_dir_slurm_logs = os.path.join(results_dir, "slurm_logs")
            command_line_l.append(f'--results_dir {results_dir} {command_line_arguments}')
            # generate dirs for every experiment
            self.generate_resul_dirs([results_dir, exp_dir_slurm_files, exp_dir_slurm_logs])
            for j in range(0, self._n_chained_jobs):
                if self._n_chained_jobs == 1:
                    exec_path = self.save_slurm(exp_dir_slurm_files, exp_dir_slurm_logs, command_line_l, str(i).zfill(len(str(len(experiment_list_chunked)))), j, self._duration)
                else:
                    duration = Launcher._to_duration(self._hours_per_job[j], 0, 0) if j < self._n_chained_jobs - 1 else Launcher._to_duration(self._hours_per_job[j], self._minutes, self._seconds)
                    exec_path = self.save_slurm(exp_dir_slurm_files, exp_dir_slurm_logs, command_line_l, str(i).zfill(len(str(len(experiment_list_chunked)))), j, duration)
                # add only first job in the loop to list, rest of the jobs will be started by in a chain
                if j == 0:
                    slurm_files_path_l.append(exec_path)

        # Launch slurm files in parallel
        for slurm_file_path in slurm_files_path_l:
            command = f"sbatch {slurm_file_path}"
            if test:
                print(command)
            else:
                os.system(command)

    def _run_local(self, test):
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._exp_file)
        experiment = module.experiment

        if test:
            self._test_experiment_local()
        else:
            def experiment_wrapper(params):
                try:
                    experiment(**params)
                except Exception:
                    print("Experiment failed with parameters:")
                    print(params)
                    traceback.print_exc()

            params_dict = get_experiment_default_params(experiment)

            Parallel(n_jobs=self._n_exps_in_parallel)(delayed(experiment_wrapper)(deepcopy(params))
                                                      for params in self._generate_exp_params(params_dict))

    def _run_sequential(self, test):
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._exp_file)
        experiment = module.experiment

        if test:
            self._test_experiment_local()
        else:
            default_params_dict = get_experiment_default_params(experiment)

            for params in self._generate_exp_params(default_params_dict):
                try:
                    experiment(**params)
                except Exception:
                    print("Experiment failed with parameters:")
                    print(params)
                    traceback.print_exc()

    def _check_experiments_results_directories(self):
        """
        Check if the results directory produced for each experiment clash.
        """
        for exp in self._experiment_list:
            results_dir = self._generate_results_dir(self._exp_dir_local, exp)
            # Check if the results directory already exists.
            if results_dir in self._results_dir_l:
                # Terminate to prevent overwriting the results directory.
                raise ResultsDirException(exp, results_dir)
            self._results_dir_l.append(results_dir)

    def _test_experiment_local(self):
        for exp, results_dir in zip(self._experiment_list, self._results_dir_l):
            for i in range(self._n_seeds):
                params = str(exp).replace('{', '(').replace('}', '').replace(': ', '=').replace('\'', '')
                if params:
                    params += ', '
                print('experiment' + params + 'seed=' + str(i) + ', results_dir=' + results_dir + ')')

    def _generate_results_dir(self, results_dir, exp, n=6):
        for key, value in exp.items():
            if key.endswith('__'):
                if self._compact_dirs:
                    subfolder = str(value)
                else:
                    subfolder = key + '_' + str(value).replace(' ', '')
                subfolder = subfolder.replace('/', '-')  # avoids creating subfolders if there is a slash in the name
                results_dir = os.path.join(results_dir, subfolder)
        return results_dir

    def _generate_exp_params(self, default_params_dict):
        seeds = np.arange(self._n_seeds)
        for exp in self._experiment_list:
            params_dict = deepcopy(default_params_dict)
            exp_new_without_underscore = self.remove_last_underscores_dict(exp)
            params_dict.update(exp_new_without_underscore)
            params_dict['results_dir'] = self._generate_results_dir(self._exp_dir_local, exp)
            for seed in seeds:
                params_dict['seed'] = int(seed)
                yield params_dict
    
    def generate_resul_dirs(self, dirs):
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    @staticmethod
    def remove_last_underscores_dict(exp_dict):
        exp_dict_new = copy(exp_dict)
        for key, value in exp_dict.items():
            if key.endswith('__'):
                exp_dict_new[key[:-2]] = value
                del exp_dict_new[key]
        return exp_dict_new

    @staticmethod
    def _convert_to_command_line(exp):
        command_line = ''
        for key, value in exp.items():
            new_command = '--' + key + ' '

            if isinstance(value, list):
                new_command += ' '.join(map(str, value)) + ' '
            else:
                new_command += str(value) + ' '

            command_line += new_command

        # remove last space
        command_line = command_line[:-1]

        return command_line

    @staticmethod
    def _to_duration(hours, minutes, seconds):
        h = "0" + str(hours) if hours < 10 else str(hours)
        m = "0" + str(minutes) if minutes < 10 else str(minutes)
        s = "0" + str(seconds) if seconds < 10 else str(seconds)

        return h + ":" + m + ":" + s

    @property
    def exp_name(self):
        return self._exp_name

    def log_dir(self, local=True):
        if local:
            return self._exp_dir_local
        else:
            return self._exp_dir_slurm

    @property
    def _exp_file_path(self):
        module = import_module(self._exp_file)
        return module.__file__


def get_experiment_default_params(func):
    signature = inspect.signature(func)
    defaults = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            defaults[k] = v.default
    return defaults


def translate_experiment_params_to_argparse(parser, func):
    annotation_to_argparse = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': None
    }
    arg_experiments = parser.add_argument_group('Experiment')
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if k not in ['seed', 'results_dir']:
            if v.default is not inspect.Parameter.empty:
                if v.annotation.__name__ in annotation_to_argparse:
                    if v.annotation.__name__ == 'bool':
                        arg_experiments.add_argument(f"--{str(k)}", type=lambda x: bool(strtobool(x)),
                                                     nargs='?', const=v.default, default=v.default)
                    elif v.annotation.__name__ == 'list':
                        arg_experiments.add_argument(f"--{str(k)}", nargs='+')
                    else:
                        arg_experiments.add_argument(f"--{str(k)}", type=annotation_to_argparse[v.annotation.__name__])
                else:
                    raise NotImplementedError(f'{v.annotation.__name__} not found in annotation_to_argparse.')
    return parser


def add_launcher_base_args(parser):
    arg_default = parser.add_argument_group('Default')
    arg_default.add_argument('--seed', type=int)
    arg_default.add_argument('--results_dir', type=str)
    return parser


def has_kwargs(func):
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if v.kind == v.VAR_KEYWORD:
            return True

    return False


def string_to_primitive(string):
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            try:
                # boolean
                return bool(strtobool(string))
            except ValueError:
                return string


def parse_unknown_args(unknown):
    kwargs = dict()

    key_idxs = [i for i, arg in enumerate(unknown) if arg.startswith('--')]

    if len(key_idxs) > 0:
        key_n_args = [key_idxs[i+1] - 1 - key_idxs[i] for i in range(len(key_idxs)-1)]
        key_n_args.append(len(unknown) - 1 - key_idxs[-1])

        for i, idx in enumerate(key_idxs):
            key = unknown[idx][2:]
            n_args = key_n_args[i]
            if n_args > 1:
                values = list()

                for v in unknown[idx+1:idx + 1 + n_args]:
                    values.append(string_to_primitive(v))

                kwargs[key] = values

            elif n_args == 1:
                kwargs[key] = string_to_primitive(unknown[idx+1])

    return kwargs


def parse_args(func):
    parser = argparse.ArgumentParser()

    parser = translate_experiment_params_to_argparse(parser, func)

    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_experiment_default_params(func))

    if has_kwargs(func):
        args, unknown = parser.parse_known_args()
        kwargs = parse_unknown_args(unknown)

        args = vars(args)
        args.update(kwargs)

        return args
    else:
        args = parser.parse_args()
        return vars(args)


def run_experiment(func, args=None):
    if not args:
        args = parse_args(func)

    func(**args)



if __name__ == '__main__':

    parse_hydra_configs_and_launch_exp()
