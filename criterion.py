import torch
import torch.nn.functional as F
import torch.distributions


def neg_log_likelihood(args, info: dict, pd_sdf: torch.Tensor, pd_sdf_std: torch.Tensor,
                       gt_sdf: torch.Tensor, **kwargs):
    """
    Negative log likelihood of gt data under predicted gaussian distribution.
    """
    if args.enforce_minmax:
        gt_sdf = torch.clamp(gt_sdf, -args.clamping_distance, args.clamping_distance)
        pd_sdf = torch.clamp(pd_sdf, -args.clamping_distance, args.clamping_distance)

    pd_dist = torch.distributions.Normal(loc=pd_sdf.squeeze(), scale=pd_sdf_std.squeeze())
    sdf_loss = -pd_dist.log_prob(gt_sdf.squeeze()).sum() / info["num_sdf_samples"]
    return {
        'll': sdf_loss
    }


def reg_loss(args, info: dict, latent_vecs: torch.Tensor, **kwargs):
    l2_size_loss = torch.sum(torch.norm(latent_vecs, dim=1))
    reg_loss = min(1, info["epoch"] / 100) * l2_size_loss / info["num_sdf_samples"]
    return {
        'reg': reg_loss * args.code_reg_lambda
    }
