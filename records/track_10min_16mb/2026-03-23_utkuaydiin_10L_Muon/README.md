# 10L Muon + Int5/Int6 PTQ + BigramHash + SmearGate (15.46 MB)

- Architecture: 448-dim, 10 layers, mixed Int5/Int6 post-training quant
- Optimizer: Muon + AdamW hybrid, WD=0.04, SWA @ ~40%
- Training: 10 min on 8×H100, global 5.5% prune
- Final size: 15.46 MB after zstd-22
- Pure modeling track
