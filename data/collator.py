import torch



def data_collator(inputs):
    batch_data = torch.concat([inp[0] for inp in inputs], dim=0)
    batch_label = torch.concat([inp[1] for inp in inputs], dim=0)
    noise = torch.randn(batch_data.shape)
    return {
        "noise": noise,
        "label": batch_label,
        "vision": batch_data,
    }