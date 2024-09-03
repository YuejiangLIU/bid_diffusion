import torch
import pdb

def euclidean_distance(src, tar, reduction='mean'):
    # B, (N), T, D
    diff = src - tar
    dist = torch.norm(diff, dim=-1)
    if reduction == 'mean':
        return dist.mean(dim=-1)
    elif reduction == 'none':
        return dist

def coverage_distance(src, tar, penalty=1000.0):
    B, S, _ = src.shape
    _, T, _ = tar.shape

    # Compute all pairwise Euclidean distances
    dist_matrix = euclidean_distance(tar.unsqueeze(2), src.unsqueeze(1), 'none')

    # Prepare to track the minimum distance for each target and the indices of matches
    min_distances = torch.full((B, T), float('inf'), device=src.device)
    matched_indices = torch.full((B, T), -1, dtype=torch.long, device=src.device)
    last_matched_indices = torch.full((B,), -1, dtype=torch.long, device=src.device)

    for t in range(T):
        for s in range(S):
            valid_mask = (s > last_matched_indices[:, None]).squeeze()
            valid_distances = torch.where(valid_mask, dist_matrix[:, t, s], float('inf'))

            min_values, min_idxs = torch.min(valid_distances.unsqueeze(-1), dim=1)
            min_mask = (min_values < min_distances[:, t])

            min_distances[:, t] = torch.where(min_mask, min_values, min_distances[:, t])
            matched_indices[:, t] = torch.where(min_mask, min_idxs + s, matched_indices[:, t])

            # Update last matched indices
            last_matched_indices = torch.where(min_mask, s * torch.ones_like(last_matched_indices), last_matched_indices)

        # Apply penalties where no match was found
        no_match = (matched_indices[:, t] == -1)
        min_distances[:, t] = torch.where(no_match, torch.tensor(penalty, device=src.device), min_distances[:, t])

    # Calculate the final distances
    final_distances = min_distances.sum(dim=1)

    # print('Matched Indices:', matched_indices)
    return final_distances
