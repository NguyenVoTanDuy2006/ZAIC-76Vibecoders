"""Policy-gradient objective on actual sampled IDs, with matching action masks."""

from __future__ import annotations

import torch


def generated_mask(tokens, eos_ids):
    eos = torch.zeros_like(tokens, dtype=torch.bool)
    for value in eos_ids:
        eos |= tokens.eq(value)
    return (eos.long().cumsum(-1) - eos.long()).eq(0)


def sequence_log_probabilities(
    logits, prompt_length, generated, eos_ids, forbidden_ids
):
    if generated.ndim != 2 or generated.shape[1] == 0 or prompt_length < 1:
        raise ValueError("expected nonempty generated token IDs and prompt")
    mask = generated_mask(generated, eos_ids)
    forbidden = torch.zeros_like(generated, dtype=torch.bool)
    for value in forbidden_ids:
        forbidden |= generated.eq(value)
    if (forbidden & mask).any():
        raise ValueError("sample contains a forbidden structural token")
    scores = logits[
        :, prompt_length - 1 : prompt_length + generated.shape[1] - 1, :
    ].float()
    if scores.shape[:2] != generated.shape:
        raise ValueError("causal logits do not align with sampled IDs")
    if forbidden_ids:
        banned = torch.tensor(forbidden_ids, device=scores.device)
        scores = scores.index_fill(-1, banned, float("-inf"))
    chosen = scores.log_softmax(-1).gather(-1, generated.unsqueeze(-1)).squeeze(-1)
    sums = torch.where(mask, chosen, torch.zeros_like(chosen)).sum(-1)
    return sums, mask


def scst_loss(sequence_logp, sampled_reward, greedy_reward):
    advantage = (sampled_reward - greedy_reward).detach().to(sequence_logp)
    if not torch.isfinite(advantage).all():
        raise ValueError("nonfinite semantic reward")
    return -(advantage * sequence_logp).mean()


def mixed_loss(ce, scst, weight=0.1):
    return ce + weight * scst
