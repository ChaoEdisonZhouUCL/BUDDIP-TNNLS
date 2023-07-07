import numpy as np


# define the adaptive loss weight
def calc_loss_weight(
    loss_weights,
    edip_downgamma,
    adip_downgamma,
    upgamma,
    epoch,
    epoch_gap=100,
    verbose=False,
    min_gamma=1e-3,
    max_gamma=1e3,
):
    (
        EDIP_mseLoss_Weight,
        ADIP_mseLoss_Weight,
        BUDIP_mseLoss_Weight,
        EDIP_angleLoss_Weight,
        ADIP_angleLoss_Weight,
        BUDIP_angleLoss_Weight,
    ) = loss_weights
    if epoch % epoch_gap == 0 and epoch != 0:
        EDIP_mseLoss_Weight *= edip_downgamma
        EDIP_angleLoss_Weight *= edip_downgamma
        ADIP_mseLoss_Weight *= adip_downgamma
        ADIP_angleLoss_Weight *= adip_downgamma
        BUDIP_mseLoss_Weight /= upgamma
        BUDIP_angleLoss_Weight /= upgamma

        EDIP_mseLoss_Weight = np.clip(EDIP_mseLoss_Weight, min_gamma, max_gamma)
        EDIP_angleLoss_Weight = np.clip(EDIP_angleLoss_Weight, min_gamma, max_gamma)
        ADIP_mseLoss_Weight = np.clip(ADIP_mseLoss_Weight, min_gamma, max_gamma)
        ADIP_angleLoss_Weight = np.clip(ADIP_angleLoss_Weight, min_gamma, max_gamma)
        BUDIP_mseLoss_Weight = np.clip(BUDIP_mseLoss_Weight, min_gamma, max_gamma)
        BUDIP_angleLoss_Weight = np.clip(BUDIP_angleLoss_Weight, min_gamma, max_gamma)

        if verbose:
            print(
                "EDIP_mseLoss_Weight=%f, ADIP_mseLoss_Weight=%f, BUDIP_mseLoss_Weight=%f, EDIP_angleLoss_Weight=%f, ADIP_angleLoss_Weight=%f, BUDIP_angleLoss_Weight=%f "
                % (
                    EDIP_mseLoss_Weight,
                    ADIP_mseLoss_Weight,
                    BUDIP_mseLoss_Weight,
                    EDIP_angleLoss_Weight,
                    ADIP_angleLoss_Weight,
                    BUDIP_angleLoss_Weight,
                )
            )

    return (
        EDIP_mseLoss_Weight,
        ADIP_mseLoss_Weight,
        BUDIP_mseLoss_Weight,
        EDIP_angleLoss_Weight,
        ADIP_angleLoss_Weight,
        BUDIP_angleLoss_Weight,
    )
