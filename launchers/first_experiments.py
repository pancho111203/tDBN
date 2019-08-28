from simplelib.launcher import generate_launcher, launch
import os
import fire
import logging

logging.basicConfig(level=logging.INFO)
task_config_filepath = os.path.join(os.path.dirname(__file__), 'first_experiments.json')
def _get_command_config(config, output=None):
    code = '/home/franciscorubin/Projects/TFM/tDBN/scripts/train.py'
    mode = 'train'
    config_path = '/home/franciscorubin/Projects/TFM/tDBN/configs/{}.config'.format(config)
    if output is None:
        output = config
    output_dir = '/home/franciscorubin/Projects/TFM/tDBN/results/{}'.format(output)
    stdout = output_dir + '/out.txt'
    stderr = output_dir + '/err.txt'

    command = 'python -u {} {} --config_path={} --model_dir={}'.format(code, mode, config_path, output_dir)
    return {
        'command': command,
        'stdout': stdout,
        'stderr': stderr
    }

def setup():
    exps = ['car_tDBN_bv_1_noDA', 'car_tDBN_bv_2_noDA', 'car_tDBN_vef_1_noDA', 'car_tDBN_vef_2_noDA']
    configs = [_get_command_config(exp) for exp in exps]

    generate_launcher(configs, task_config_filepath)

def run():
    launch(task_config_filepath, experiment_indices=None, run_errored=True, usable_gpus=[0, 1], max_parallel_workers=2, verbose=False, cwd=None)

if __name__ == '__main__':
    fire.Fire()