"""
Description:
    this is the module provides losses functions。

"""


import numpy as np
import torch
import torch.nn.functional as F


def angle_distance_loss_torch():
    def myloss(y_true, y_pred):
        dot_product = (y_true * y_pred).sum(dim=-1)
        l2_norms = torch.norm(y_true, dim=-1) * torch.norm(y_pred, dim=-1) + 1e-8
        cosine_similarity = dot_product / l2_norms

        # Clamp the cosine similarity to the range [-1, 1] to avoid NaN values
        eps = 1e-7
        cosine_similarity = torch.clamp(
            cosine_similarity, min=-1.0 + eps, max=1.0 - eps
        )
        add_loss = torch.acos(cosine_similarity) * 180.0 / 3.14

        return torch.mean(add_loss)

    return myloss


def RMSE_loss_torch(y_true, y_pred):
    MSE = torch.mean(torch.square(y_true - y_pred), dim=-1)
    rmse = torch.sqrt(MSE)
    return torch.mean(rmse)


#########################################################################################################
# HSI unmixing metrics
#########################################################################################################
def NMSE_metric(y_true, y_pred):
    MSE = np.mean(np.square(y_true - y_pred))
    norm = np.mean(np.square(y_true - np.mean(y_true, axis=0, keepdims=True)))
    return 10 * np.log10(MSE / norm)


def RMSE_metric(y_true, y_pred):
    """

    :param y_true: (No_Pixels, No_Endm)
    :param y_pred: (No_Pixels, No_Endm)
    :return:
    """
    MSE = np.mean(np.square(y_true - y_pred), axis=-1)
    rmse = np.sqrt(MSE)
    # averaged over pixels
    return np.mean(rmse)


def MAE_metric(y_true, y_pred):
    MAE = np.abs(y_true - y_pred) * 100
    return np.mean(MAE)


def angle_distance_metric(y_true, y_pred, verbose=False):
    """

    :param y_true: (No_Endm, No_Bands)
    :param y_pred: (No_Endm, No_Bands)
    :return:
    """

    dot_product = np.sum(y_true * y_pred, axis=-1)
    l2_norms = np.linalg.norm(y_true, axis=-1) * np.linalg.norm(y_pred, axis=-1) + 1e-8
    cosine_similarity = dot_product / l2_norms

    # Clamp the cosine similarity to the range [-1, 1] to avoid NaN values
    eps = 1e-7
    cosine_similarity = np.clip(cosine_similarity, a_min=-1.0 + eps, a_max=1.0 - eps)
    AAD = np.arccos(cosine_similarity) * 180.0 / np.pi

    if verbose:
        print(f"angle distance is: {AAD}")

    # returned value is averaged over different number of endmembers
    if verbose:
        return AAD, np.mean(AAD)
    else:
        return np.mean(AAD)


def information_divergence_metric(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-8)
    y_true = np.maximum(y_true, 1e-8)
    divergence_1 = np.divide(y_true, y_pred)
    # divergence_1 = np.maximum(divergence_1, 1e-8)
    divergence_1 = np.sum(np.multiply(np.log(divergence_1), y_true), axis=-1)

    divergence_2 = np.divide(y_pred, y_true)
    # divergence_2 = np.maximum(divergence_2, 1e-8)

    divergence_2 = np.sum(np.multiply(np.log(divergence_2), y_pred), axis=-1)

    AID = divergence_1 + divergence_2

    return np.mean(AID)
