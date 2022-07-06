import numpy as np
import matplotlib.pyplot as plt

import torch
from IPython.display import clear_output

from utils import random_noise_vectors, tensor2image, smooth1d


def discriminator_step(models, optimizers, X, Z, y_ones, y_zeros, criterion):
    D = models['discriminator']
    G = models['generator']
    
    D_loss_real = criterion(D(X), y_ones) 
    D_loss_fake = criterion(D(G(Z).detach()), y_zeros)

    optimizers['discriminator'].zero_grad()
    D_loss = (D_loss_real+D_loss_fake)/2
    D_loss.backward() # может быть стоит делать backward отдельно сначала на D_loss_real, потом на D_loss_fake (GAN Hacks)
    optimizers['discriminator'].step()

    return D_loss_real.item(), D_loss_fake.item()


def generator_step(models, optimizers, y_ones, criterion, generate_noise, generator_learning_steps=1):
    D = models['discriminator']
    G = models['generator']
    
    G_step_losses = []
    for i in range(generator_learning_steps):
        # Может быть и не стоит на обучении генератора генерировать новый шум,
        # а использовать тот же, что и при обучении дискриминатора, как здесь:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        Z = generate_noise()

        # Modified Minimax Loss
        # see: https://developers.google.com/machine-learning/gan/loss
        G_loss = criterion(D(G(Z)), y_ones)

        optimizers['generator'].zero_grad()
        G_loss.backward()
        optimizers['generator'].step()

        G_step_losses.append(G_loss.item())
        
    G_loss = sum(G_step_losses)/len(G_step_losses)
        
    return G_loss


def train_step_graph(generated_images, D_epoch_losses_real, D_epoch_losses_fake, G_epoch_losses, 
                     examples_suptitle_text='', losses_suptitle_text='', losses_smooth_window=25):
    num_examples = len(generated_images)
    
    fig, axs = plt.subplots(1, num_examples, figsize=(num_examples*2, num_examples))
    for i in range(num_examples):
        plt.suptitle(examples_suptitle_text)
        axs[i].imshow(generated_images[i])
        axs[i].set_title('Generated Image')
        axs[i].axis('off')
    fig.tight_layout(pad=2)
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(smooth1d(np.array(D_epoch_losses_real), losses_smooth_window), label='D_losses_real')
    ax.plot(smooth1d(np.array(D_epoch_losses_fake), losses_smooth_window), label='D_losses_fake')
    ax.plot(smooth1d(np.array(G_epoch_losses), losses_smooth_window), label='G_losses')
    ax.legend()
    plt.suptitle(losses_suptitle_text)
    plt.show()


def train(models, latent_size, criterion, optimizers, dataloader, dataset_mean, dataset_std, 
          epochs=20, label_smooth=0.1, generator_learning_steps=1, 
          graph_show_interval=10, losses_smooth_window=25, device='cpu'):
    
    if label_smooth<0 or label_smooth>1:
        raise ValueError
    
    # генерация векторов, на которых будет генерироваться картинка во время обучения
    num_examples = 4
    example_noise = random_noise_vectors(num_examples, latent_size, device)

    # списки лоссов за все эпохи
    D_losses_real = []
    D_losses_fake = []
    G_losses = []

    for epoch in range(epochs):
        # списки лоссов за все батчи в эпохе
        D_epoch_losses_real = []
        D_epoch_losses_fake = []
        G_epoch_losses = []

        for batch_num, X_batch in enumerate(dataloader):
            batch_size = X_batch.shape[0]
            Z_batch = random_noise_vectors(batch_size, latent_size, device)
            X_batch = X_batch.to(device)
            y_ones = torch.ones(batch_size, 1, device=device)
            
            # label smoothing
            y_ones_smooth = y_ones*(1-label_smooth)
            y_zeros_smooth = torch.zeros(batch_size, 1, device=device)+label_smooth

            # Discriminator training
            # -----------------------------------
            D_loss_real, D_loss_fake = discriminator_step(
                models, optimizers, X_batch, Z_batch, y_ones_smooth, y_zeros_smooth, criterion
            )
            D_epoch_losses_real.append(D_loss_real)
            D_epoch_losses_fake.append(D_loss_fake)
            # -----------------------------------

            # Generator training
            # -----------------------------------
            generate_noise = lambda: random_noise_vectors(batch_size, latent_size, device)
            G_loss = generator_step(
                models, optimizers, y_ones, criterion, generate_noise, generator_learning_steps
            )
            G_epoch_losses.append(G_loss)
            # -----------------------------------

            # Example images and losses graph 
            # -----------------------------------
            if batch_num % graph_show_interval == 0:
                losses_suptitle_text = f"Epoch: {epoch+1}, {batch_num+1}/{len(dataloader)}"
                examples_suptitle_text = f"D_loss_real {D_loss_real:.5f}, D_loss_fake {D_loss_fake:.5f}, G_loss {G_loss:.5f}"
                generated_images = [tensor2image(genim, dataset_mean, dataset_std) for genim in models['generator'](example_noise)]
                
                clear_output(wait=True)
                train_step_graph(
                    generated_images, D_epoch_losses_real, D_epoch_losses_fake, G_epoch_losses, 
                    examples_suptitle_text, losses_suptitle_text, losses_smooth_window
                )
            # -----------------------------------

        D_epoch_loss_real = sum(D_epoch_losses_real)/len(dataloader)
        D_epoch_loss_fake = sum(D_epoch_losses_fake)/len(dataloader)
        G_epoch_loss = sum(G_epoch_losses)/len(dataloader)
        D_losses_real += [D_epoch_loss_real]
        D_losses_fake += [D_epoch_loss_fake]
        G_losses += [G_epoch_loss]
        
    return D_losses_real, D_losses_fake, G_losses
                        