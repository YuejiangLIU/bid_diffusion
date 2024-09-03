"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import pdb

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-e', '--noise', default=0.0)
@click.option('-p', '--perturb', default=0.0)
@click.option('-ah', '--ahorizon', default=8)
@click.option('-t', '--ntest', default=200)
@click.option('-s', '--sampler', required=True)
@click.option('-n', '--nsample', default=1)
@click.option('-m', '--nmode', default=1)
@click.option('-k', '--decay', default=0.9)
@click.option('-r', '--reference', default=None)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, noise, perturb, ahorizon, ntest, sampler, nsample, nmode, decay, reference, device):
    if os.path.exists(output_dir):
        print(f"Output path {output_dir} already exists and will be overwrited.")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load reference
    if reference:
        try:
            payload = torch.load(open(reference, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg, output_dir=output_dir)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            weak = workspace.model
            if cfg.training.use_ema:
                weak = workspace.ema_model
            weak.n_action_steps = ahorizon
            device = torch.device(device)
            weak.to(device)
            weak.eval()
            print('Loaded weak model')
        except Exception as e:
            weak = None
            print('Skipped weak model')

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # turn off video
    cfg.task.env_runner['n_train_vis'] = 0
    cfg.task.env_runner['n_test_vis'] = 2
    cfg.task.env_runner['n_train'] = 1
    cfg.task.env_runner['n_test'] = ntest
    cfg.task.env_runner['n_action_steps'] = ahorizon
    cfg.task.env_runner['test_start_seed'] = 20000

    policy.n_action_steps = ahorizon

    print("\nEvaluation setting:")
    print("env:")
    try:
        print(f"  delay_horizon = {cfg.task.env_runner.n_latency_steps: <19} act_horizon = {cfg.task.env_runner.n_action_steps: <15} obsv_horizon = {cfg.task.env_runner.n_obs_steps}")
    except Exception as e:
        print(f"  delay_horizon = {float('nan'): <19} act_horizon = {cfg.task.env_runner.n_action_steps: <15} obsv_horizon = {cfg.task.env_runner.n_obs_steps}")
    print("policy:")
    print(f"  pred_horizon = {policy.horizon: <20} act_horizon = {policy.n_action_steps: <15} obsv_horizon = {policy.n_obs_steps}")

    # run eval
    if perturb > 0:
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir,
            max_steps=400,
            perturb_level=perturb)
    else:
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)

    # set sampler
    env_runner.set_sampler(sampler, nsample, nmode, noise, decay)
    if reference and weak:
        env_runner.set_reference(weak)

    runner_log = env_runner.run(policy)
    print(f"train: {runner_log['train/mean_score']}         test: {runner_log['test/mean_score']}")

    # dump log to json
    json_log = dict()
    json_log['checkpoint'] = checkpoint
    for key, value in runner_log.items():
        if 'video' not in key:
            json_log[key] = value
    if sampler == 'random':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}.json')
    elif sampler == 'lowvar':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}.json')
    elif sampler == 'ema':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{decay}.json')
    elif sampler == 'contrast':
        rname = reference.split('epoch=')[1].replace('.json', '').split('-')[0]
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}_{rname}.json')
    elif sampler == 'positive':
        rname = reference.split('epoch=')[1].replace('.json', '').split('-')[0]
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}_{rname}.json')
    elif sampler == 'negative':
        rname = reference.split('epoch=')[1].replace('.json', '').split('-')[0]
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}_{rname}.json')
    elif sampler == 'coherence':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{decay}.json')
    elif sampler == 'bid':
        rname = reference.split('epoch=')[1].replace('.json', '').split('-')[0]   
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{nmode}_{decay}_{rname}.json')
    elif sampler == 'cma':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{decay}.json')
    elif sampler == 'warmstart':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}.json')
    elif sampler == 'wma':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{decay}.json')
    elif sampler == 'warmcoherence':
        out_path = os.path.join(output_dir, f'eval_{ntest}_{noise}_{sampler}_{policy.horizon}-{ahorizon}_{nsample}_{decay}.json')
    else:
        pass
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
