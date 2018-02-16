import sys, os, argparse, torch
import models, pipeline


val_set = pipeline.IntrinsicDataset('dataset/output/','motorbike_val',['normals', 'lights', 'shading'], size_per_dataset=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, num_workers=4, shuffle=False)
shader = models.Shader().cuda()
# pipeline.visualize_shader(shader, val_loader, "saved/a.png",save_raw=True)
val_losses = pipeline.visualize_decomposer(shader, val_loader, "saved/a.png", 1,save_raw=True)
