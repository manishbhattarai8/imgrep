import torch.nn.functional as F
import torch

def contrastive_loss(image_features, text_features, temperature=0.07):
    logits = image_features @ text_features.T
    labels = torch.arange(logits.shape[0]).to(logits.device)
    logits = logits / temperature
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2