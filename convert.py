import torch
from pathlib import Path


# enc_jittor_path = Path("/home/huangjh/shared-home/implicit-slam/di-checkpoints/default/encoder_1.pth.tar")
# enc_pth_path = Path("/home/huangjh/shared-home/implicit-slam/public/di-public-code/ckpt/default/encoder_300.pth.tar")
#
# enc_pth_weight = torch.load(enc_pth_path)["model_state"]
# enc_jt_weight = torch.load(enc_jittor_path)["model_state"]

dec_jittor_path = Path("/home/huangjh/shared-home/implicit-slam/di-checkpoints/default/model_1.pth.tar")
dec_pth_path = Path("/home/huangjh/shared-home/implicit-slam/public/di-public-code/ckpt/default/model_300.pth.tar")

dec_pth_weight = torch.load(dec_pth_path)["model_state"]
dec_jt_weight = torch.load(dec_jittor_path)["model_state"]


for i, v in dec_jt_weight.items():
    print(i, v.size())
