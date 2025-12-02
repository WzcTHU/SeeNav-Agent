import torch
import numpy as np
import uuid
from collections import defaultdict
from verl import DataProto

def compute_step_rewards(batch: DataProto):
    all_returns = batch.non_tensor_batch['raw_step_rewards'].astype(np.float32)
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

def compute_srgpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   step_group_size: int,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_std_norm",
                                   intask: bool = False,
                                   intraj: bool = False
                                   ):
    """
    Compute the advantages for SRGPO.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages.
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    if intask and not intraj:
        step_group_uids = build_step_group(step_rewards=step_rewards, step_group_size=step_group_size, index=index, traj_index=None)
    if not intask and intraj:
        step_group_uids = build_step_group(step_rewards=step_rewards, step_group_size=step_group_size, index=None, traj_index=traj_index)
    else:
        step_group_uids = build_step_group(step_rewards=step_rewards, step_group_size=step_group_size, index=None, traj_index=None)
    # Compute step relative advantages.
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages.
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores

def build_step_group(step_rewards: torch.Tensor = None, step_group_size: int = 16, index: np.array = None, traj_index = None): # index is episode-level group uid
    bsz = step_rewards.shape[0]
    step_group_uids = np.empty(bsz, dtype=object)

    if index is None and traj_index is None:
        indices = np.arange(bsz)
        np.random.shuffle(indices)
        if step_group_size > bsz or step_group_size == -1:      # use full batch as step group
            step_group_size = bsz
        for i in range(0, bsz, step_group_size):
            group_indices = indices[i:i+step_group_size]
            uid = str(uuid.uuid4())
            step_group_uids[group_indices] = uid
    
    elif index is not None and traj_index is None:
        unique_group_ids = np.unique(index)
        for group_id in unique_group_ids:
            group_mask = (index == group_id)
            same_group_sample_indices = np.where(group_mask)[0]
            shuffled_indices = np.copy(same_group_sample_indices)
            np.random.shuffle(shuffled_indices)
            num_samples = len(shuffled_indices)
            actual_group_size = step_group_size
            if step_group_size > num_samples or step_group_size == -1:
                actual_group_size = num_samples
            for i in range(0, num_samples, actual_group_size):
                sub_group_indices = shuffled_indices[i:i+actual_group_size]
                uid = str(uuid.uuid4())
                step_group_uids[sub_group_indices] = uid

    elif index is None and traj_index is not None:
        unique_traj_ids = np.unique(traj_index)
        for traj_id in unique_traj_ids:
            traj_mask = (traj_index == traj_id)
            same_traj_sample_indices = np.where(traj_mask)[0]
            shuffled_indices = np.copy(same_traj_sample_indices)
            np.random.shuffle(shuffled_indices)
            num_samples = len(shuffled_indices)
            actual_group_size = step_group_size
            if step_group_size > num_samples or step_group_size == -1:
                actual_group_size = num_samples
            for i in range(0, num_samples, actual_group_size):
                sub_group_indices = shuffled_indices[i:i+actual_group_size]
                uid = str(uuid.uuid4())
                step_group_uids[sub_group_indices] = uid

    else:
        raise ValueError(f"index and traj_index cannot be both not None")

    return step_group_uids

def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages
