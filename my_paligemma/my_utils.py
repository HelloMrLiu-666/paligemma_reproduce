import torch
from torch import nn


def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Args:
        probs:(batch, num_posi1?,vocab_size)
    Returns:
        next_token:(batch, num_posi1?)
    """
    batch,_,vocab_size=probs.shape
    #(batch, num_posi1?,vocab_size)
    probs_sort, probs_idx=torch.sort(probs, dim=-1, descending=True)
    #(batch, num_posi1?,vocab_size) no shape change
    probs_sum=torch.cumsum(probs_sort, dim=-1)
    #(batch, num_posi1?,vocab_size) no shape change, boolean mask ,true means delete, false means keep
    mask=probs_sum-probs_sort>p
    #boolean indexing, and assign true elements to be probability of 0
    probs_sort[mask]=0.0
    #(batch, num_posi1?,vocab_size)->(batch, num_posi1?,1)
    sorted_probs_sum=torch.sum(probs_sort, dim=-1, keepdim=True)
    #normalize the probs, (batch, num_posi1?,vocab_size)&(batch, num_posi1?,1)->(batch, num_posi1?,vocab_size)
    probs_sort=probs_sort/sorted_probs_sum
    #sample from normalized probs:final shape is (batch,num_posi1?,1)
        #(batch, num_posi1?,vocab_size) view->(batch*num_posi1?,vocab_size) sample->(batch*num_posi1?,1) view->(batch,num_posi1?,1)
    next_token=torch.multinomial(probs_sort.view(-1,vocab_size), num_samples=1,generator=None).contiguous().view(batch,-1,1)
    #(batch, num_posi1?,vocab_size) & (batch,num_posi1?,1)->(batch,num_posi1?,1) squeeze-> (batch, num_posi1?)
    next_token=torch.gather(probs_idx, -1, next_token).squeeze(-1)
    return next_token