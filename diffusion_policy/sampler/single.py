import torch
from diffusion_policy.sampler.metric import euclidean_distance, coverage_distance

import pdb
torch.set_printoptions(precision=1, sci_mode=False)

def coherence_sampler(policy, prior, obs_dict, num_sample=10, beta=0.5):
    """
    Sample an action from a policy that preserves coherence with a prior.

    Args:
        policy: a policy network to predict sequences of actions
        prior: the prediction made in the previous time step
        obs_dict: dictionary containing observations at the current time step
        num_sample (int, optional): number of samples to generate
        beta (float, optional): weight decay factor for coherence

    Returns:
        dict: a selected dictionary of actions 
    """
    if prior is None:
        return policy.predict_action(obs_dict)

    # pre-process
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    for key in obs_dict.keys():
        if key == 'prior':
            continue
        obs_dict_batch[key] = obs_dict[key].unsqueeze(1).repeat(1, num_sample, 1, 1).view(B * num_sample, OH, OD)

    # predict
    action_dict_batch = policy.predict_action(obs_dict_batch)

    # post-process
    AH, PH, AD = action_dict_batch['action'].shape[1], action_dict_batch['action_pred'].shape[1], action_dict_batch['action_pred'].shape[2]
    action_dict_batch['action'] = action_dict_batch['action'].view(B, num_sample, AH, AD)
    action_dict_batch['action_pred'] = action_dict_batch['action_pred'].view(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_dict_batch:
        action_dict_batch['action_obs_pred'] = action_dict_batch['action_obs_pred'].view(B, num_sample, AH, OD)
    if 'obs_pred' in action_dict_batch:
        action_dict_batch['obs_pred'] = action_dict_batch['obs_pred'].view(B, num_sample, PH, OD)

    # distance measure
    start_overlap = policy.n_obs_steps - 1
    end_overlap = prior.shape[1]
    dist_raw = euclidean_distance(action_dict_batch['action_pred'][:, :, start_overlap:end_overlap], prior.unsqueeze(1)[:, :, start_overlap:], reduction='none')

    weights = torch.tensor([beta**i for i in range(end_overlap-start_overlap)]).to(dist_raw.device)
    weights = weights / weights.sum()
    dist_weighted = dist_raw * weights.view(1, 1, end_overlap-start_overlap)
    dist = dist_weighted.sum(dim=2)

    # sample selection
    _, cross_index = dist.sort(descending=False)
    index = cross_index[:, 0]

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_dict_batch.keys():
        action_dict[key] = action_dict_batch[key][range_tensor, index]

    return action_dict

def ema_sampler(policy, prior, obs_dict, beta):
    action_dict = policy.predict_action(obs_dict)
    if prior is not None:
        # frame matching
        if policy.oa_step_convention:
            start = policy.n_obs_steps - 1
        else:
            start = policy.n_obs_steps
        end = start + policy.n_action_steps
        assert (action_dict['action'] == action_dict['action_pred'][:,start:end]).all().item()
        # ema update
        CH = prior.shape[1]
        action_dict['action_pred'][:,:CH] = prior * beta + action_dict['action_pred'][:,:CH] * (1. - beta)
        action_dict['action'] = action_dict['action_pred'][:,start:end]
    return action_dict

def cma_sampler(policy, prior, obs_dict, num_sample=10, beta1=0.75, beta2=0.95):
    if prior is None:
        return policy.predict_action(obs_dict)

    # pre-process
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    for key in obs_dict.keys():
        obs_dict_batch[key] = obs_dict[key].unsqueeze(1).repeat(1, num_sample, 1, 1).reshape(B * num_sample, OH, OD)

    # predict
    action_dict_batch = policy.predict_action(obs_dict_batch)

    # post-process
    AH, PH, AD = action_dict_batch['action'].shape[1], action_dict_batch['action_pred'].shape[1], action_dict_batch['action_pred'].shape[2]
    action_dict_batch['action'] = action_dict_batch['action'].reshape(B, num_sample, AH, AD)
    action_dict_batch['action_pred'] = action_dict_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_dict_batch:
        action_dict_batch['action_obs_pred'] = action_dict_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_dict_batch:
        action_dict_batch['obs_pred'] = action_dict_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    # distance measure
    CH = prior.shape[1]
    dist_raw = euclidean_distance(action_dict_batch['action_pred'][:, :, :CH], prior.unsqueeze(1), reduction='none')

    weights = torch.tensor([beta2**i for i in range(CH)]).to(dist_raw.device)
    weights = weights / weights.sum()
    dist_weighted = dist_raw * weights.view(1, 1, CH)
    dist = dist_weighted.sum(dim=2)

    # sample selection
    _, cross_index = dist.sort(descending=False)
    index = cross_index[:, 0]

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_dict_batch.keys():
        action_dict[key] = action_dict_batch[key][range_tensor, index]

    # frame matching
    if policy.oa_step_convention:
        start = policy.n_obs_steps - 1
    else:
        start = policy.n_obs_steps
    end = start + policy.n_action_steps
    assert (action_dict['action'] == action_dict['action_pred'][:,start:end]).all().item()

    # ema update
    action_dict['action_pred'][:,:CH] = prior * beta1 + action_dict['action_pred'][:,:CH] * (1. - beta1)
    action_dict['action'] = action_dict['action_pred'][:,start:end]

    return action_dict
