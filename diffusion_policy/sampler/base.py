import torch

def preprocess_obs_batch(obs_dict):
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    for key in obs_dict.keys():
        obs_dict_batch[key] = obs_dict[key].unsqueeze(1).repeat(1, num_sample, 1, 1).reshape(B * num_sample, OH, OD)
    return obs_dict_batch

def postprocess_action_batch(action_dict_batch):
    AH, PH, AD = action_dict_batch['action'].shape[1], action_dict_batch['action_pred'].shape[1], action_dict_batch['action_pred'].shape[2]
    action_dict_batch['action'] = action_dict_batch['action'].reshape(B, num_sample, AH, AD)
    action_dict_batch['action_pred'] = action_dict_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_dict_batch:
        action_dict_batch['action_obs_pred'] = action_dict_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_dict_batch:
        action_dict_batch['obs_pred'] = action_dict_batch['obs_pred'].reshape(B, num_sample, PH, OD)
    return action_dict_batch

def slice_action_batch(action_dict_batch):
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_dict_batch.keys():
        action_dict[key] = action_dict_batch[key][range_tensor, index]
    return action_dict