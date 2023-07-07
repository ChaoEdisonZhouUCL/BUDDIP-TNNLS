"""
Description:
this file provides data pre-processing funcs.

"""


import numpy as np
from scipy.io import loadmat


# {code}
def load_JasperRidge_data():
    data_path = "../Data/jasperRidge2_R198.mat"
    label_path = "../Data/end4.mat"
    data_params = {
        "NO_Bands": None,
        "NO_Endms": None,
        "NO_DATA": None,
        "img_size": None,
    }

    data = loadmat(data_path)
    label = loadmat(label_path)

    x = data["Y"].T
    Norm = data["maxValue"].astype(np.float64)
    hsi_data = x / Norm

    abundance = label["A"].T
    abundance = np.clip(abundance, 0.0, 1.0)

    endm_sig = label["M"]

    data_params["NO_Bands"] = x.shape[1]
    data_params["NO_Endms"] = abundance.shape[1]
    data_params["NO_DATA"] = x.shape[0]
    data_params["img_size"] = (
        data["nRow"][0, 0].astype(np.int32),
        data["nCol"][0, 0].astype(np.int32),
    )

    return hsi_data, endm_sig, abundance, data_params
