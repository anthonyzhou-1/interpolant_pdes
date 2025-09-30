import torch
from torch import Tensor
import math 
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator


def generate_palette(hex_color, n_colors=5, saturation="light"):
    if saturation == "light":
        palette = sns.light_palette(hex_color, n_colors=n_colors, as_cmap=False)
    elif saturation == "dark":
        palette = sns.dark_palette(hex_color, n_colors=n_colors, as_cmap=False)
    return palette


color_dict = {"wasserstein": "#cc241d", "mmd": "#eebd35", "c2st": "#458588"}

# Mostly from https://github.com/mackelab/labproject

def johnson_lindenstrauss(x, k):
    # x is a tensor of shape (n_samples, n_features)
    gaussian_matrix = torch.randn(k, x.shape[-1]) # (k, n_features)
    x = 1/math.sqrt(k) * torch.matmul(gaussian_matrix, x.permute(1, 0))  # (k, n_samples)
    x = x.permute(1, 0)  # (n_samples, k)
    return x

# Sliced Wasserstein distances

def sliced_wasserstein_distance(
    encoded_samples: Tensor,
    distribution_samples: Tensor,
    num_projections: int = 50,
    p: int = 2,
    device: str = "cpu",
) -> Tensor:
    """
    Sliced Wasserstein distance between encoded samples and distribution samples.
    Note that the SWD does not converge to the true Wasserstein distance, but rather it is a different proper distance metric.

    Args:
        encoded_samples (torch.Tensor): tensor of encoded training samples
        distribution_samples (torch.Tensor): tensor drawn from the prior distribution
        num_projection (int): number of projections to approximate sliced wasserstein distance
        p (int): power of distance metric
        device (torch.device): torch device 'cpu' or 'cuda' gpu

    Return:
        torch.Tensor: Tensor of wasserstein distances of size (num_projections, 1)
    """

    # check input (n,d only)
    assert len(encoded_samples.size()) == 2, "Real samples must be 2-dimensional, (n,d)"
    assert len(distribution_samples.size()) == 2, "Fake samples must be 2-dimensional, (n,d)"

    embedding_dim = distribution_samples.size(-1)

    projections = rand_projections(embedding_dim, num_projections).to(device)

    encoded_projections = encoded_samples.matmul(projections.transpose(-2, -1))

    distribution_projections = distribution_samples.matmul(projections.transpose(-2, -1))

    wasserstein_distance = (
        torch.sort(encoded_projections.transpose(-2, -1), dim=-1)[0]
        - torch.sort(distribution_projections.transpose(-2, -1), dim=-1)[0]
    )

    wasserstein_distance = torch.pow(torch.abs(wasserstein_distance), p)

    return torch.pow(torch.mean(wasserstein_distance, dim=(-2, -1)), 1 / p)


def rand_projections(embedding_dim: int, num_samples: int):
    """
    This function generates num_samples random samples from the latent space's unti sphere.r

    Args:
        embedding_dim (int): dimention of the embedding
        sum_samples (int): number of samples

    Return :
        torch.tensor: tensor of size (num_samples, embedding_dim)
    """

    ws = torch.randn((num_samples, embedding_dim))
    projection = ws / torch.norm(ws, dim=-1, keepdim=True)
    return projection

# C2ST (Classifier Two-Sample Test)

def c2st_nn(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    z_score: bool = True,
    activation = "relu",
    clf_kwargs = {},
) -> Tensor:
    r"""
    Return accuracy of MLP classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Training of the `MLPClassifier` from `sklearn.neural_network` is performed with
    N-fold cross-validation [3]. Before both samples are ingested, they are normalized
    (z scored) under the assumption that each dimension in X follows a normal distribution,
    i.e. the mean(X) is subtracted from X and this difference is divided by std(X)
    for every dimension.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_score` method in this module.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        z_score: Z-scoring using X, i.e. mean and std deviation of X is used to normalize Y, i.e. Y=(Y - mean)/std
        activation: Activation function for the hidden layer
        clf_kwargs: Additional kwargs for `MLPClassifier`

    Return:
        torch.tensor containing the mean accuracy score over the test sets
        from cross-validation

    Example:
    ``` py
    > c2st_nn(X,Y)
    [0.51904464] #X and Y likely come from the same PDF or ensemble
    > c2st_nn(P,Q)
    [0.998456] #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """

    # the default configuration
    clf_class = MLPClassifier
    ndim = X.shape[-1]
    defaults = {
        "activation": activation,
        "hidden_layer_sizes": (min(10 * ndim, 100), min(10 * ndim, 100)),
        "max_iter": 1000,
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 50,
    }
    defaults.update(clf_kwargs)

    scores_ = c2st_scores(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        metric=metric,
        z_score=z_score,
        noise_scale=None,
        verbosity=0,
        clf_class=clf_class,
        clf_kwargs=defaults,
    )

    scores = np.mean(scores_).astype(np.float32)
    value = torch.from_numpy(np.atleast_1d(scores))
    return value

def c2st_scores(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    z_score: bool = True,
    noise_scale = None,
    verbosity: int = 0,
    clf_class = MLPClassifier,
    clf_kwargs = {},
) -> Tensor:
    r"""
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    This function performs training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used which
    is recommended based on the study performed in [4].
    This can be changed using <clf_class>. This class is used in the following
    fashion:

    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```
    Further configuration of the classifier can be performed using <clf_kwargs>.
    If you like to provide a custom class for training, it has to satisfy the
    internal requirements of `sklearn.model_selection.cross_val_score`.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross validation
        n_folds: Number of folds to use for cross validation
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        z_score: Z-scoring using X, i.e. mean and std deviation of X is used to normalize Y, i.e. Y=(Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with standard deviation <noise_scale> to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictionary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier

    Return:
        np.ndarray containing the calculated <metric> scores over the test set
        folds from cross-validation

    Example:
    ``` py
    > c2st_scores(X,Y)
    [0.51904464,0.5309201,0.4959452,0.5487709,0.50682926]
    #X and Y likely come from the same PDF or ensemble
    > c2st_scores(P,Q)
    [0.998456,0.9982912,0.9980476,0.9980488,0.99805826]
    #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    clf_kwargs["random_state"] = seed
    clf = clf_class(**clf_kwargs)

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity)

    return scores

# MMD (Maximum Mean Discrepancy)

def rbf_kernel(x, y, bandwidth):
    dist = torch.cdist(x, y)
    return torch.exp(-(dist**2) / (2.0 * bandwidth**2))


def polynomial_kernel(x, y, degree, bias):
    return (x @ y.t() + bias) ** degree


def linear_kernel(x, y):
    return x @ y.t()


def energy_kernel(x, y):
    x_norm = torch.linalg.norm(x, dim=-1)
    y_norm = torch.linalg.norm(y, dim=-1)
    return x_norm[:, None] + y_norm[None, :] - torch.cdist(x, y)


def median_heuristic(x, y):
    return torch.median(torch.cdist(x, y))

def compute_rbf_mmd(x, y, bandwidth=1.0):
    x_kernel = rbf_kernel(x, x, bandwidth)
    y_kernel = rbf_kernel(y, y, bandwidth)
    xy_kernel = rbf_kernel(x, y, bandwidth)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

def compute_rbf_mmd_median_heuristic(x, y):
    # https://arxiv.org/pdf/1707.07269.pdf
    bandwidth = median_heuristic(x, y)
    return compute_rbf_mmd(x, y, bandwidth)

def compute_rbf_mmd_auto(x, y, bandwidth=1.0):
    dim = x.shape[1]
    x_kernel = rbf_kernel(x, x, dim * bandwidth)
    y_kernel = rbf_kernel(y, y, dim * bandwidth)
    xy_kernel = rbf_kernel(x, y, dim * bandwidth)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

def compute_polynomial_mmd(x, y, degree=2, bias=0):
    x_kernel = polynomial_kernel(x, x, degree, bias)
    y_kernel = polynomial_kernel(y, y, degree, bias)
    xy_kernel = polynomial_kernel(x, y, degree, bias)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

def compute_linear_mmd_naive(x, y):
    x_kernel = linear_kernel(x, x)
    y_kernel = linear_kernel(y, y)
    xy_kernel = linear_kernel(x, y)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

def compute_linear_mmd(x, y):
    delta = torch.mean(x, 0) - torch.mean(y, 0)
    return torch.norm(delta, 2) ** 2

def compute_energy_mmd(x, y):
    x_kernel = energy_kernel(x, x)
    y_kernel = energy_kernel(y, y)
    xy_kernel = energy_kernel(x, y)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd


# plotting
def plot_distances(
    num_dists,
    distances,
    errors,
    metric_name,
    dataset_name,
    ax=None,
    label=None,
    **kwargs,
):
    """Plot the scaling of a metric with increasing dimensionality."""
    if ax is None:
        plt.plot(
            num_dists,
            distances,
            label=metric_name if label is None else label,
            **kwargs,
        )
        plt.fill_between(
            num_dists,
            distances - errors,
            distances + errors,
            alpha=0.2,
            color="black" if kwargs.get("color") is None else kwargs.get("color"),
        )
        plt.xlabel("Timesteps")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} across timesteps for {dataset_name}")
    else:
        ax.plot(
            num_dists,
            distances,
            label=metric_name if label is None else label,
            **kwargs,
        )
        ax.fill_between(
            num_dists,
            distances - errors,
            distances + errors,
            alpha=0.2,
            color="black" if kwargs.get("color") is None else kwargs.get("color"),
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(
            metric_name, color="black" if kwargs.get("color") is None else kwargs.get("color")
        )
        return ax