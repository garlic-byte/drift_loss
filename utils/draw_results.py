import os

import torch
import math
import matplotlib.pyplot as plt
from transformers import TrainerCallback

class DrawResults(TrainerCallback):
    def __init__(self, model, input_channels, img_size, draw_step, plot_dir, device, num_classes, plot_nums=100):
        self.model = model
        self.input_channels = input_channels
        self.img_size = img_size
        self.draw_step = draw_step
        self.plot_dir = plot_dir
        self.device = device
        self.num_classes = num_classes
        self.plot_nums = plot_nums
        self.last_validate_step = 0

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def on_step_end(self, args, state, control, **kwargs):
        # Varify whether draw results
        current_step = state.global_step
        if current_step % self.draw_step == 0 and current_step != self.last_validate_step:
            self.last_validate_step = current_step
            self.plot_outputs()

    def plot_outputs(self):
        self.model.eval()
        noise = torch.randn(self.plot_nums, self.input_channels, self.img_size, self.img_size).to(self.device)
        labels = torch.arange(self.num_classes).to(self.device)
        labels = labels.unsqueeze(-1).repeat(1, math.ceil(self.plot_nums/self.num_classes)).flatten()
        with torch.no_grad():
            outputs = self.model(noise, labels)
    
        fig, axes = plt.subplots(math.ceil(self.plot_nums / 10), 10, figsize=(15, 15))
        fig.suptitle(f"Epoch_{self.last_validate_step}", fontsize=16)
        axes = axes.flatten()
        for index in range(self.plot_nums):
            img_np = outputs[index].detach().cpu().permute(1, 2, 0).numpy()
            img_np = (img_np / 2) + 0.5
            img_np = img_np.clip(0, 1).squeeze(axis=-1)
            axes[index].imshow(img_np, cmap='gray')
            axes[index].axis('off')
    
        plt.tight_layout()
        save_path = f"{self.plot_dir}/epoch_{self.last_validate_step}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
