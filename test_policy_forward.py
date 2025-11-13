#!/usr/bin/env python3
import torch
from train_asymmetric_rl_optimized import AsymmetricPolicyNetwork

print("PyTorch version", torch.__version__)

policy = AsymmetricPolicyNetwork(
    macro_features=48,
    micro_features=48,
    macro_embedding_dim=256,
    micro_hidden_dim=256,
    num_actions=3
)

policy = policy.float()
policy.train()

macro = torch.randn(8,48)
micro = torch.randn(8,48)
position = torch.zeros(8)
cash_ratio = torch.ones(8)

print("Running forward...")
output,_ = policy(macro, micro, position, cash_ratio)
print("Output shape", output.shape)
