import torch
import matplotlib.pyplot as plt

def entropy_spectrum(vorticity_batch: torch.Tensor):
    """
    Compute the azimuthally averaged entropy spectrum of 2D vorticity fields using PyTorch and einsum.

    Parameters
    ----------
    vorticity_batch : torch.Tensor
        Tensor of shape (batch, nx, ny, 1), real-valued vorticity fields.

    Returns
    -------
    spectra : torch.Tensor
        Tensor of shape (batch, k_max+1), the entropy spectrum for each sample.
    k_vals : torch.Tensor
        Tensor of shape (k_max+1,), the corresponding wavenumber bins.
    """
    batch, nx, ny, _ = vorticity_batch.shape
    device = vorticity_batch.device

    # Remove channel dimension for FFT
    vorticity = vorticity_batch[..., 0]  # (batch, nx, ny)

    # Compute FFT and power spectrum
    fft2 = torch.fft.fft2(vorticity, norm='forward')  # (batch, nx, ny)
    power = fft2.abs() ** 2  # (batch, nx, ny)

    # Frequency grid
    kx = torch.fft.fftfreq(nx, d=1.0, device=device).reshape(-1, 1).repeat(1, ny)
    ky = torch.fft.fftfreq(ny, d=1.0, device=device).reshape(1, -1).repeat(nx, 1)
    k_mag = torch.sqrt((kx * nx) ** 2 + (ky * ny) ** 2)
    k_int = torch.round(k_mag).to(torch.int64)  # integer wavenumbers

    # Flatten spatial dimensions
    power_flat = power.reshape(batch, -1)             # (batch, nx*ny)
    k_int_flat = k_int.reshape(-1)                    # (nx*ny,)
    k_max = int(k_int_flat.max().item())
    k_vals = torch.arange(k_max + 1, device=device)   # (k_max+1,)

    # Build a mask matrix of shape (k_max+1, nx*ny): 1 if point belongs to shell k
    mask = (k_int_flat[None, :] == k_vals[:, None]).float()  # (k_max+1, nx*ny)

    # Use einsum to sum power within each shell: result is (batch, k_max+1)
    spectra = torch.einsum('bk,nk->bn', power_flat, mask)

    return spectra, k_vals


def plot_result_2d(u, rec=None, n_t=5, path=None, cmap="twilight_shifted"):
    '''
    Plot the results of a 2D PDE model.
    Args:
        u (torch.Tensor): ground truth velocity field, shape (b, nt, nx, ny, c)
        rec (torch.Tensor): predicted velocity field, shape (b, nt, nx, ny, c), can be None
        n_t (int): number of timesteps to plot
        path (str): path to save the plot, if None, the plot will not be saved
    Returns:
        None
    '''

    u = u[0, ..., 0].detach().cpu() # (nt nx ny)
    rec = rec[0, ..., 0].detach().cpu() if rec is not None else None
        
    vmin = torch.min(u)
    vmax = torch.max(u)

    if n_t == 1:
        u_downs = u # (1, nx, ny)
        n_skip = 1
    else:
        n_skip = u.shape[0] // n_t 
        u_downs = u[::n_skip]

    if rec is not None:
        fig, ax = plt.subplots(n_t, 2, figsize=(8, 4*n_t))

        if n_t == 1:
            rec_downs = rec
            ax = [ax]
        else:
            rec_downs = rec[::n_skip]

        for j in range(2):
            for i in range(n_t):
                ax[i][j].set_axis_off()
                
                if j == 0:
                    velocity = u_downs[i] 
                else:
                    velocity = rec_downs[i]

                im = ax[i][j].imshow(velocity, vmin=vmin, vmax=vmax, cmap=cmap)
                ax[i][j].title.set_text(f'Timestep {i*n_skip}')
            ax[0][j].title.set_text(f'Ground Truth' if j == 0 else f'Prediction')
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))
        ax = [ax] if n_t == 1 else ax

        for i in range(n_t):
            ax[i].set_axis_off()
            velocity = u_downs[i] 

            im = ax[i].imshow(velocity, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[i].title.set_text(f'Timestep {i*n_skip}')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)

def plot_entropy_2d(u, rec=None, n_t=5, path=None):
    '''
    Plot the results of a 2D PDE model.
    Args:
        u (torch.Tensor): ground truth velocity field, shape (b, nt, nx, ny, c)
        rec (torch.Tensor): predicted velocity field, shape (b, nt, nx, ny, c), can be None
        n_t (int): number of timesteps to plot
        path (str): path to save the plot, if None, the plot will not be saved
    Returns:
        None
    '''

    u = u[0, ..., 0].detach().cpu() # (nt nx ny)
    rec = rec[0, ..., 0].detach().cpu() if rec is not None else None

    if n_t == 1:
        u_downs = u # (1, nx, ny)
        n_skip = 1
    else:
        n_skip = u.shape[0] // n_t 
        u_downs = u[::n_skip]

    if rec is not None:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))

        if n_t == 1:
            rec_downs = rec
            ax = [ax]
        else:
            rec_downs = rec[::n_skip]

        for i in range(n_t):
            sample_true = u_downs[i].unsqueeze(-1).unsqueeze(0)
            sample_pred = rec_downs[i].unsqueeze(-1).unsqueeze(0)

            entropy_true, k_vals = entropy_spectrum(sample_true)
            entropy_pred, _ = entropy_spectrum(sample_pred)

            ax[i].loglog(k_vals.cpu(), entropy_true[0].cpu(), label='Ground Truth', color='blue')
            ax[i].loglog(k_vals.cpu(), entropy_pred[0].cpu(), label='Prediction', color='orange', linestyle='--')
            ax[i].title.set_text(f'Timestep {i*n_skip}')
            ax[i].legend()
        
    else:
        fig, ax = plt.subplots(n_t, 1, figsize=(4, 4*n_t))
        ax = [ax] if n_t == 1 else ax

        for i in range(n_t):
            sample_true = u_downs[i].unsqueeze(-1).unsqueeze(0)

            entropy_true, k_vals = entropy_spectrum(sample_true)

            ax[i].loglog(k_vals.cpu(), entropy_true[0].cpu(), label='Ground Truth', color='blue')
            ax[i].title.set_text(f'Timestep {i*n_skip}')
            ax[i].legend()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.close(fig)