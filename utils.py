import os
import numpy as np

import torch


save_folder = 'saved'


def tensor2image(inp, dataset_mean, dataset_std):
    """Преобразует PyTorch тензоры для использования в np.imshow"""
    out = inp.cpu().detach().numpy().transpose((1, 2, 0))
    mean = np.array(dataset_mean)
    std = np.array(dataset_std)
    out = std * out + mean

    return np.clip(out, 0, 1)


# Сохранение и загрузка моделей
def save_models(D, G, D_optim, G_optim, losses, name):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    state = {
        'D_model': D.state_dict(),
        'G_model': G.state_dict(),
        'D_optim': D_optim.state_dict(),
        'G_optim': G_optim.state_dict(),
        'losses': losses
        }
    torch.save(state, os.path.join(save_folder, f"{name}.pth"))

    
def load_models(D, G, D_optim, G_optim, name, device='cuda'):
    state = torch.load(os.path.join(save_folder, f"{name}.pth"), map_location=device)
    D.load_state_dict(state['D_model'])
    G.load_state_dict(state['G_model'])
    D_optim.load_state_dict(state['D_optim'])
    G_optim.load_state_dict(state['G_optim'])
    return state['losses']


# генерация вектора шума из нормального распределения mean=0, std=1
def random_noise_vectors(n, dim, device='cpu'):
    return torch.normal(torch.zeros(n, dim), torch.ones(n, dim)).to(device)


def smooth1d(data, window_width):
    """Сглаживает данные усреднением по окну размера window_width"""
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width