import argparse
import torch
import os
import yaml
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from common.loss import _weight_for_latitude_vector_without_poles

def plot_loss(losses, filename, key=None):
    # losses in shape [t]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Loss')
    ax.set_title(f'{key} loss over time')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_result_climate(y_pred, y, filename, num_t=6, cmap='twilight_shifted'):
    # y in shape [t h w], y_pred in shape [t h w]

    t_total, h, w = y_pred.shape

    dt = 0
    if num_t != 1:
        dt = t_total // num_t
        y_pred = y_pred[::dt]
        y = y[::dt]

    fig, axs = plt.subplots(2, num_t, figsize=(num_t*6, 6))

    vmin = y.min()
    vmax = y.max()

    for i in range(num_t):
        if num_t == 1:
            im0 = axs[0].imshow(y[i], vmin=vmin, vmax=vmax,cmap=cmap)
            im1 = axs[1].imshow(y_pred[i], vmin=vmin, vmax=vmax, cmap=cmap)

            # set the title
            axs[0].set_title(f"True t={(i+1)*dt}")
            axs[1].set_title(f"Pred t={(i+1)*dt}")
        else:
            im0 = axs[0][i].imshow(y[i], vmin=vmin, vmax=vmax,cmap=cmap)
            im1 = axs[1][i].imshow(y_pred[i], vmin=vmin, vmax=vmax, cmap=cmap)

            # set the title
            axs[0][i].set_title(f"True t={(i+1)*dt}")
            axs[1][i].set_title(f"Pred t={(i+1)*dt}")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im0, cax=cbar_ax)
    # save the figure
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_climatological_bias(pred, target, save_path=None):
    # pred, target in shape nlat nlon
    bias = pred - target
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    vmin = target.min()
    vmax = target.max()
    
    bias_min = bias.min()
    bias_max = bias.max()

    bias_scale = max(abs(bias_min), abs(bias_max))

    im0 = axs[0].imshow(pred, cmap='twilight_shifted', vmin=vmin, vmax=vmax)
    axs[0].set_title('Predicted')
    fig.colorbar(im0, ax=axs[0], orientation='horizontal')

    im1 = axs[1].imshow(target, cmap='twilight_shifted', vmin=vmin, vmax=vmax)
    axs[1].set_title('Target')
    fig.colorbar(im1, ax=axs[1], orientation='horizontal')

    im2 = axs[2].imshow(bias, cmap='bwr', vmin=-bias_scale, vmax=bias_scale)
    axs[2].set_title('Bias (Predicted - Target)')
    fig.colorbar(im2, ax=axs[2], orientation='horizontal')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_crps(crps, title, t=120, save_path=None):
    plt.figure()
    plt.plot(crps)
    plt.xticks(np.arange(0, t+1, 24), np.arange(0, t//4+1, 6))
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('CRPS')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_ssr(ssr, title, t=120, save_path=None):
    plt.figure()
    plt.plot(ssr)
    plt.xticks(np.arange(0, t+1, 24), np.arange(0, t//4+1, 6))
    plt.xlabel('Forecast lead time (days)')
    plt.ylabel('SSR')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_spectrum(pred, target, path, num_t = 6):
    # pred and target in shape (t, nlat, nlon)
    if pred.shape[0] == 1: # assume t is trivial
        pred = torch.from_numpy(pred).squeeze()
        target = torch.from_numpy(target).squeeze()
        nlat = pred.shape[0]
        nlon = pred.shape[1]

        k_x_pred, power_spectrum_pred = zonal_averaged_power_spectrum(pred, nlon=nlon, nlat=nlat)
        k_x_target, power_spectrum_target = zonal_averaged_power_spectrum(target, nlon=nlon, nlat=nlat)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(k_x_pred, power_spectrum_pred, label='Predicted', color='blue')
        ax.plot(k_x_target, power_spectrum_target, label='Target', color='orange')
        ax.set_xlabel('Zonal wavenumber')
        ax.set_ylabel('Power Spectrum')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Zonal Averaged Power Spectrum')
        ax.legend()
        plt.savefig(path, dpi=300)
        plt.close()
    else:
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
        t_total = pred.shape[0]
        dt = t_total // num_t
        pred = pred[::dt]
        target = target[::dt]

        nlat = pred.shape[1]
        nlon = pred.shape[2]

        fig, axs = plt.subplots(1, num_t, figsize=(num_t*6, 6))

        for i in range(num_t):
            k_x_pred, power_spectrum_pred = zonal_averaged_power_spectrum(pred[i], nlon=nlon, nlat=nlat)
            k_x_target, power_spectrum_target = zonal_averaged_power_spectrum(target[i], nlon=nlon, nlat=nlat)

            axs[i].plot(k_x_pred, power_spectrum_pred, label='Predicted', color='blue')
            axs[i].plot(k_x_target, power_spectrum_target, label='Target', color='orange')
            axs[i].set_xlabel('Zonal wavenumber')
            axs[i].set_ylabel('Power Spectrum')
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_title(f'Zonal Averaged Power Spectrum at t={(i+1)*dt}')
            axs[i].legend()

        plt.savefig(path, dpi=300)
        plt.close()

def zonal_averaged_power_spectrum(field,
                                  nlon=128,
                                  nlat=64):
    """
    This function calculates the zonal averaged power spectrum of a given field. It is designed to work with xarray DataArrays or Datasets that have 'lat', 'lon', and optionally 'time' dimensions. The function first transposes the dimensions to ensure 'lat' and 'lon' are the first two dimensions, then performs a Fast Fourier Transform (FFT) along the 'lon' axis to compute the power spectrum. The power spectrum is then averaged over 'lat' and 'time' (if present) to produce the zonal averaged power spectrum.

    Parameters:
    - field, tensor of shape (nlat, nlon)
    """

    field = torch.transpose(field, 0, 1)  # nlon, nlat

    ###########################################################################################
    field_fft = torch.fft.rfft(field, dim=0, norm='forward') # Convention used: the first Fourier coefficient is the mean of the field

    # Compute the power spectrum (squared magnitude of Fourier coefficients)
    power_spectrum = torch.abs(field_fft)**2

    # Define the zonal wavenumbers
    nx = nlon
    k_x = torch.fft.fftfreq(nx, d=1/nx)

    # Only take the positive frequencies (or the first half if using real FFT)
    k_x = k_x[:nx//2]
    power_spectrum = power_spectrum[:nx//2]
    # count the positive frequencies twice except for the first one (zero frequency), because the FFT of a real function is symmetric
    power_spectrum[1:] *= 2
    # multiply by a factor cos(pi latitude[i] / 180) in axis 1
    lat_end = (nlat-1)*(360/nlon) / 2
    latitude = np.linspace(-lat_end, lat_end, nlat)
    weights = _weight_for_latitude_vector_without_poles(latitude)
    weights = torch.from_numpy(weights)
    weights = weights.view(1, -1)
    power_spectrum *= weights 

    power_spectrum_avg = power_spectrum.mean(axis=1)

    return k_x, power_spectrum_avg

def assemble_input(surface_input, multilevel_input):
    multilevel_collapsed = rearrange(multilevel_input, 'b nlat nlon nlevel c -> b nlat nlon (nlevel c)')
    model_input = torch.cat([surface_input, multilevel_collapsed], dim=-1) # b nlat nlon (c + nlevel*c) or b nface nside nside (c + nlevel*c)
    return model_input

def assemble_grid_params(constants, yearly_constants, t):
    # constants in shape b nlat nlon c, yearly_constants in shape b t nlat nlon c
    yearly_constants_t = yearly_constants[:, t] # b nlat nlon c 
    grid_params = torch.cat([constants, yearly_constants_t], dim=-1) # b nlat nlon (c + c)
    return grid_params

def assemble_scalar_params(day_of_year, hour_of_day, t):
    # day of year in shape b t, hour of day in shape b t
    return torch.cat([day_of_year[:, t].unsqueeze(1), hour_of_day[:, t].unsqueeze(1)], dim=1) # b 2

def disassemble_input(assembled_input, num_levels=13, num_surface_channels=8):
    surface_input = assembled_input[..., :num_surface_channels]
    multilevel_input = rearrange(assembled_input[..., num_surface_channels:], 'b nlat nlon (nlevel c) -> b nlat nlon nlevel c', nlevel=num_levels)
    return surface_input, multilevel_input

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self