########################################################################################################################
# Project: nonlinear blind unmixing using DIP techniques.

# In this file, we provide the nonlinear output layer for nonlinear Blind unmixing.
########################################################################################################################

import torch
import torch.nn as nn


class Nonlinear_Mixing_layer_torch(nn.Module):
    def __init__(self, nonlinear_model):
        super(Nonlinear_Mixing_layer_torch, self).__init__()
        self.nonlinear_model = nonlinear_model

    def forward(self, inputs):
        """
        :param inputs:
        if nonlinear_model = 'FM':
            [E, A], where A is the abundance estimation, E is the endm estiamtion.
                        shape(A)=(1, no_Endms, img_row, img_col),
                        shape(E)=(1, no_Bands, no_Endms)
        if nonlinear_model='GBM':
        [E, A, Gamma], where A is the abundance estimation, E is the endm estiamtion, Gamma is the nonlinear coefficents.
                        shape(A)=(1, no_Endms, img_row, img_col),
                        shape(E)=(1, no_Bands, no_Endms),
                        shape(Gamma)=(1, no_Endms^2, no_Bands, no_Endms)

        :param kwargs:
        :return:
        """

        if self.nonlinear_model == "FM":
            assert isinstance(inputs, list)
            E, A = inputs

            # linear mixing part
            linear_output = torch.tensordot(E, A[0], dims=([2], [0]))

            # nonlinear mixing part
            nonlinear_output = 0.0

            NO_Endms = E.shape[-1]
            for i in range(NO_Endms - 1):
                for j in range(i + 1, NO_Endms):
                    abu = torch.unsqueeze(torch.mul(A[:, i, :, :], A[:, j, :, :]), 1)
                    endm = torch.unsqueeze(torch.mul(E[:, :, i], E[:, :, j]), 2)
                    nonlinear_output += torch.tensordot(endm, abu[0], dims=([2], [0]))

        elif self.nonlinear_model == "GBM":
            assert isinstance(inputs, list)
            E, A, Gamma = inputs

            # linear mixing part
            linear_output = torch.tensordot(E, A[0], dims=([2], [0]))

            # nonlinear mixing part
            nonlinear_output = 0.0

            NO_Endms = E.shape[-1]
            for i in range(NO_Endms - 1):
                for j in range(i + 1, NO_Endms):
                    abu = torch.unsqueeze(torch.mul(A[:, i, :, :], A[:, j, :, :]), 1)
                    abu = torch.unsqueeze(
                        torch.mul(abu[:, 0, :, :], Gamma[:, i * NO_Endms + j, :, :]), 1
                    )
                    endm = torch.unsqueeze(torch.mul(E[:, :, i], E[:, :, j]), 2)
                    nonlinear_output += torch.tensordot(endm, abu[0], dims=([2], [0]))

        else:
            raise ValueError("nonlinear_model has no value: " + self.nonlinear_model)

        return linear_output + nonlinear_output
