########################################################################################################################
# Project: nonlinear blind unmixing using DIP techniques.

# In this file, we implement the nonlinear BUDIP using ADIP, EDIP and nonlinear output layer from their corresponding
#   Modules.
########################################################################################################################
import os.path
import shutil
import sys
import time

import numpy as np
from scipy.io import loadmat, savemat

sys.path.append("..")
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Code.Modules.AdaptiveLossWeight import calc_loss_weight
from Code.Modules.ADIP import (ADIP_angle_loss_torch, ADIP_loss_torch,
                               ADIP_torch)
from Code.Modules.Data_preprocessing import load_JasperRidge_data
from Code.Modules.EDIP import (EDIP_angle_loss_torch, EDIP_loss_torch,
                               EDIP_torch)
from Code.Modules.Gamma_DIP import Gamma_DIP_torch
from Code.Modules.Losses import (RMSE_metric, angle_distance_loss_torch,
                                 angle_distance_metric)
from Code.Modules.Nonlinear_output_Layer import Nonlinear_Mixing_layer_torch
from Code.Modules.utils import (create_project_log_path, get_noise,
                                np_to_torch, plot_abundance_map, plot_Endm,
                                summary2readme, torch_to_np)
from Code.torchinfo.torchinfo import summary_net


class Nonlinear_BUDIP(nn.Module):
    def __init__(self, img_row, img_col, NO_Bands, NO_Endms, nonlinear_model):
        super(Nonlinear_BUDIP, self).__init__()
        """
            In this function, we define the nonlinear blind unmixing network using torch.
            :param img_row: number of row of HSI image
            :param img_col: number of col of HSI image
            :param NO_Bands: number of bands of HSI image
            :param NO_Endms: number of endmembers of HSI image
            :param nonlinear_model: nonlinear HSI unmixing model, e.x., FM.
            :return: net
            """

        # define EDIP
        input_shape = (NO_Bands, NO_Endms)
        output_shape = input_shape
        num_filters_down = [256]
        num_filters_up = [input_shape[0]]
        filter_size_up = [3, 3]
        filter_size_down = [3, 3]
        filter_size_skip = 1
        downsample_modes = ["stride", "stride"]
        strides = [1, 1]
        activation = "leaky_relu"
        padding = "same"
        self.E_net = EDIP_torch(
            input_shape,
            output_shape,
            num_filters_up,
            num_filters_down,
            filter_size_up,
            filter_size_down,
            filter_size_skip,
            downsample_modes,
            strides,
            activation,
            padding,
        )

        # # define ADIP
        input_shape = (NO_Endms, img_row, img_col)
        output_shape = input_shape
        num_filters_down = [32, 64]
        num_filters_up = [64, input_shape[0]]
        filter_size_up = [3, 3]
        filter_size_down = [3, 3]
        filter_size_skip = 1
        downsample_modes = ["stride", "stride"]
        strides = [1, 1]
        activation = "leaky_relu"
        padding = "same"
        self.A_net = ADIP_torch(
            input_shape,
            output_shape,
            num_filters_up,
            num_filters_down,
            filter_size_up,
            filter_size_down,
            filter_size_skip,
            downsample_modes,
            strides,
            activation,
            padding,
        )

        # output of nonlinear BUDIP
        self.nonlinear_mix_layer = Nonlinear_Mixing_layer_torch(
            nonlinear_model=nonlinear_model
        )
        self.nonlinear_model = nonlinear_model
        if self.nonlinear_model == "GBM":
            # # define Gamma_DIP
            input_shape = (NO_Endms**2, img_row, img_col)
            output_shape = input_shape
            num_filters_down = [256, 512]
            num_filters_up = [256, input_shape[0]]
            filter_size_up = [3, 3]
            filter_size_down = [3, 3]
            filter_size_skip = 1
            downsample_modes = ["stride", "stride"]
            strides = [1, 1]
            activation = "leaky_relu"
            padding = "same"
            self.Gamma_net = Gamma_DIP_torch(
                input_shape,
                output_shape,
                num_filters_up,
                num_filters_down,
                filter_size_up,
                filter_size_down,
                filter_size_skip,
                downsample_modes,
                strides,
                activation,
                padding,
            )

    def forward(self, inputs):
        if self.nonlinear_model == "GBM":
            E_input, A_input, Gamma_input = inputs
            A_output = self.A_net(A_input)
            E_output = self.E_net(E_input)
            Gamma_output = self.Gamma_net(Gamma_input)
            output = self.nonlinear_mix_layer([E_output, A_output, Gamma_output])
        else:
            E_input, A_input = inputs
            A_output = self.A_net(A_input)
            E_output = self.E_net(E_input)
            output = self.nonlinear_mix_layer([E_output, A_output])

        return E_output, A_output, output


def main():
    Endm_ext_method = "EDAA"

    Nonlinear_model = "FM"  # 'FM', 'GBM'

    ##--------------------------------------------------------------------------------------
    # 1. load data
    hsi_data, true_endm_sig, true_abundances, data_params = load_JasperRidge_data()
    log_dir = f"../results/nonlinear/{Endm_ext_method}/"

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    hsi_data = hsi_data.astype(np.float32)
    true_endm_sig = true_endm_sig.astype(np.float32)
    true_abundances = true_abundances.astype(np.float32)

    NO_Bands, NO_Endms = true_endm_sig.shape
    img_row, img_col = data_params["img_size"]
    img_row, img_col = np.int32(img_row), np.int32(img_col)

    ##--------------------------------------------------------------------------------------
    # 2. guidance generation : EDAA
    endmember_guidance = loadmat("../Data/EDAA-guidance/EDAA_endmember.mat")
    endmember_guidance = endmember_guidance["est_endmember_rescale"]
    abundance_guidance = loadmat("../Data/EDAA-guidance/EDAA_abundance.mat")
    abundance_guidance = abundance_guidance["est_abundance"]
    abundance_guidance = np.clip(abundance_guidance, 1e-6, 1 - (1e-6))

    ##--------------------------------------------------------------------------------------
    # 3. prepare inputs and outputs
    x_Abundance = np.reshape(abundance_guidance, (img_row, img_col, NO_Endms))
    x_Abundance = np.transpose(x_Abundance, axes=[2, 0, 1]).astype(np.float32)

    y_Abundance = np.reshape(true_abundances, (img_row, img_col, NO_Endms))
    y_Abundance = np.transpose(y_Abundance, axes=[2, 0, 1]).astype(np.float32)

    x_Endmember = endmember_guidance.astype(np.float32)

    y_Endmember = true_endm_sig.astype(np.float32)

    y = np.reshape(hsi_data, (img_row, img_col, NO_Bands))
    y = np.transpose(y, (2, 0, 1)).astype(np.float32)

    EDIP_Input = x_Endmember
    ADIP_Input = x_Abundance
    if Nonlinear_model == "GBM":
        Gamma_Input = (
            get_noise(NO_Endms**2, "noise", (img_row, img_col))
            .detach()
            .cpu()
            .numpy()[0, :, :, :]
        )
        data = [
            EDIP_Input,
            ADIP_Input,
            Gamma_Input,
            x_Abundance,
            y_Abundance,
            x_Endmember,
            y_Endmember,
            y,
        ]
    else:
        data = [
            EDIP_Input,
            ADIP_Input,
            x_Abundance,
            y_Abundance,
            x_Endmember,
            y_Endmember,
            y,
        ]

    ##--------------------------------------------------------------------------------------
    # 4. define train function
    def train_test_model(data, run_name, hparams, sema):
        seed = int(time.time())
        torch.manual_seed(seed)
        np.random.seed(seed)
        # choose a GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if Nonlinear_model == "GBM":
            [
                EDIP_Input,
                ADIP_Input,
                Gamma_Input,
                x_Abundance,
                y_Abundance,
                x_Endmember,
                y_Endmember,
                y,
            ] = data
            Gamma_Input_torch = np_to_torch(Gamma_Input).to(device)
        else:
            (
                EDIP_Input,
                ADIP_Input,
                x_Abundance,
                y_Abundance,
                x_Endmember,
                y_Endmember,
                y,
            ) = data
        # move the data into gpu
        EDIP_Input_torch = np_to_torch(EDIP_Input).to(device)
        ADIP_Input_torch = np_to_torch(ADIP_Input).to(device)
        x_Abundance_torch = np_to_torch(x_Abundance).to(device)
        x_Endmember_torch = np_to_torch(x_Endmember).to(device)
        y_torch = np_to_torch(y).to(device)

        # build net
        net = Nonlinear_BUDIP(
            img_row, img_col, NO_Bands, NO_Endms, nonlinear_model=Nonlinear_model
        ).to(device)

        # define loss and opt
        EDIP_MSE_criterion = EDIP_loss_torch(x_Abundance_torch, y_torch)
        EDIP_angle_criterion = EDIP_angle_loss_torch(x_Abundance_torch, y_torch)
        ADIP_MSE_criterion = ADIP_loss_torch(x_Endmember_torch, y_torch)
        ADIP_angle_criterion = ADIP_angle_loss_torch(x_Endmember_torch, y_torch)
        BUDIP_MSE_criterion = nn.MSELoss()
        BUDIP_angle_criterion = angle_distance_loss_torch()

        optimizer = optim.Adam(net.parameters(), lr=hparams["lr"])
        # ---------------------------- experiment log ----------------------------
        Readme = (
            "Try nonlinear Blind unmixing on real data.\r\n"
            + "Model invovled: Nonlinear_BUDIP.\r\n"
            + "Nonlinear Mode: "
            + Nonlinear_model
            + ".\r\n"
            + "data_params:\r\n"
            + str(data_params)
            + "\r\n"
            + run_name
            + "\r\n"
            + "seed: "
            + str(seed)
            + "\r\n"
            + "Endmember extraction method: "
            + Endm_ext_method
            + "\r\n"
            + str(hparams)
            + "\r\n"
        )

        kwargs = {
            "Readme": Readme,
        }
        (
            program_log_path,
            model_checkpoint_dir,
            tensorboard_log_dir,
            model_log_dir,
        ) = create_project_log_path(project_path=log_dir + run_name, **kwargs)

        # the SAD of endmember_guidance and the rmse of est_abu
        sad_endmwise, sad_guidance = angle_distance_metric(
            y_Endmember.T, x_Endmember.T, verbose=True
        )
        summary_str = (
            Endm_ext_method
            + " est endm SAD: %f\r\n" % (sad_guidance)
            + f"endm wise SAD : {sad_endmwise}"
        )
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        rmse_guidance = RMSE_metric(true_abundances, abundance_guidance)
        summary_str = "EDAA est abu RMSE: %f" % (rmse_guidance)
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")
        aad_guidance = angle_distance_metric(true_abundances, abundance_guidance)
        summary_str = "EDAA est abu AAD: %f" % (aad_guidance)
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        # plot the input
        plt.figure()
        plt.plot(EDIP_Input)
        plt.title("E_input: SAD=%f" % (sad_guidance))
        plt.savefig(program_log_path + "E_input.png")
        plt.close()

        plot_abundance_map(
            np.reshape(true_abundances, (img_row, img_col, NO_Endms)),
            np.reshape(abundance_guidance, (img_row, img_col, NO_Endms)),
            filepath=program_log_path + "A_Input ",
            suptitle="rmse=%f" % (rmse_guidance),
        )

        # summary network into readme.txt
        sum = summary_net(net.E_net, EDIP_Input_torch.shape, device=device)
        print(sum, file=open(program_log_path + "Readme.txt", "a"))
        sum = summary_net(net.A_net, ADIP_Input_torch.shape, device=device)
        print(sum, file=open(program_log_path + "Readme.txt", "a"))

        # define tensorboard writer
        writer = SummaryWriter(tensorboard_log_dir)

        # train
        num_epochs = hparams["epochs"]
        EDIP_mseLoss_Weight = hparams["EDIP_loss_weight"]
        ADIP_mseLoss_Weight = hparams["ADIP_loss_weight"]
        BUDIP_mseLoss_Weight = hparams["BUDIP_loss_weight"]
        EDIP_angleLoss_Weight = hparams["EDIP_angleloss_weight"]
        ADIP_angleLoss_Weight = hparams["ADIP_angleloss_weight"]
        BUDIP_angleLoss_Weight = hparams["BUDIP_angleloss_weight"]
        best_RMSE, best_AAD, best_SAD = np.inf, np.inf, np.inf
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            if Nonlinear_model == "GBM":
                E_output, A_output, output = net(
                    [EDIP_Input_torch, ADIP_Input_torch, Gamma_Input_torch]
                )
            else:
                E_output, A_output, output = net([EDIP_Input_torch, ADIP_Input_torch])

            E_mse_loss = EDIP_MSE_criterion(x_Endmember_torch, E_output)
            E_angle_loss = EDIP_angle_criterion(x_Endmember_torch, E_output)
            A_mse_loss = ADIP_MSE_criterion(x_Abundance_torch, A_output)
            A_angle_loss = ADIP_angle_criterion(x_Abundance_torch, A_output)
            BU_mse_loss = BUDIP_MSE_criterion(output, y_torch)
            BU_angle_loss = BUDIP_angle_criterion(y_torch, output)

            (
                EDIP_mseLoss_Weight,
                ADIP_mseLoss_Weight,
                BUDIP_mseLoss_Weight,
                EDIP_angleLoss_Weight,
                ADIP_angleLoss_Weight,
                BUDIP_angleLoss_Weight,
            ) = calc_loss_weight(
                loss_weights=[
                    EDIP_mseLoss_Weight,
                    ADIP_mseLoss_Weight,
                    BUDIP_mseLoss_Weight,
                    EDIP_angleLoss_Weight,
                    ADIP_angleLoss_Weight,
                    BUDIP_angleLoss_Weight,
                ],
                edip_downgamma=hparams["down_gamma"],
                adip_downgamma=hparams["down_gamma"],
                upgamma=hparams["up_gamma"],
                epoch=epoch,
                epoch_gap=hparams["gap"],
                min_gamma=hparams["min_gamma"],
                max_gamma=hparams["max_gamma"],
            )

            loss = (
                EDIP_mseLoss_Weight * E_mse_loss
                + EDIP_angleLoss_Weight * E_angle_loss
                + ADIP_mseLoss_Weight * A_mse_loss
                + ADIP_angleLoss_Weight * A_angle_loss
                + BUDIP_mseLoss_Weight * BU_mse_loss
                + BUDIP_angleLoss_Weight * BU_angle_loss
            )

            loss.backward()
            optimizer.step()

            # print training process
            if (epoch + 1) % 100 == 0 or epoch == 0:
                A_output_np = torch_to_np(A_output)
                E_output_np = torch_to_np(E_output)
                rmse = RMSE_metric(
                    y_Abundance.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                    A_output_np.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                )
                aad = angle_distance_metric(
                    y_Abundance.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                    A_output_np.transpose((1, 2, 0)).reshape(
                        img_row * img_col, NO_Endms
                    ),
                )

                sad = angle_distance_metric(y_Endmember.T, E_output_np.T)

                best_AAD = np.minimum(best_AAD, aad)
                if best_RMSE > rmse:
                    best_RMSE = rmse
                    savemat(
                        tensorboard_log_dir + "best est_abundance.mat",
                        {"est_abundance": A_output_np},
                    )
                if best_SAD > sad:
                    best_SAD = sad
                    savemat(
                        tensorboard_log_dir + "best est_endmember.mat",
                        {"est_endmember": E_output_np},
                    )
                    torch.save(net.state_dict(), model_checkpoint_dir + "model.pth")

                summary_str = (
                    "epoch %d/%d: loss = %f, EDIP_loss = %f, ADIP_loss = %f, Nonlinear_BUDIP_loss = %f, EDIP_angle_loss = %f, ADIP_angle_loss = %f, Nonlinear_BUDIP_angle_loss = %f, rmse = %f, aad = %f, sad = %f"
                    % (
                        epoch + 1,
                        num_epochs,
                        loss.item(),
                        E_mse_loss.item(),
                        A_mse_loss.item(),
                        BU_mse_loss.item(),
                        E_angle_loss.item(),
                        A_angle_loss.item(),
                        BU_angle_loss.item(),
                        rmse,
                        aad,
                        sad,
                    )
                )
                print(summary_str)
                summary2readme(summary_str, program_log_path + "Readme.txt")

            # ...log the running loss into board
            if (epoch + 1) % 100 == 0 or epoch == 0:
                writer.add_scalar("total loss", loss.item(), epoch + 1)
                writer.add_scalar("EDIP mse loss", E_mse_loss.item(), epoch + 1)
                writer.add_scalar("ADIP mse loss", E_mse_loss.item(), epoch + 1)
                writer.add_scalar(
                    "Nonlinear_BUDIP mse loss", BU_mse_loss.item(), epoch + 1
                )
                writer.add_scalar("EDIP angle loss", E_angle_loss.item(), epoch + 1)
                writer.add_scalar("ADIP angle loss", E_angle_loss.item(), epoch + 1)
                writer.add_scalar(
                    "Nonlinear_BUDIP angle loss", BU_angle_loss.item(), epoch + 1
                )
                writer.add_scalar("abu RMSE", rmse, epoch + 1)
                writer.add_scalar("abu AAD", aad, epoch + 1)
                writer.add_scalar("endm SAD", sad, epoch + 1)

                # ...log the est endm and abudance
                fig = plot_Endm(
                    y_Endmember,
                    E_output_np,
                    suptitle="epoch %d: sad= %f" % (epoch + 1, sad),
                )
                writer.add_figure("est endm", fig, epoch + 1, close=True)
                plt.close("all")

                fig = plot_abundance_map(
                    y_Abundance.transpose((1, 2, 0)),
                    A_output_np.transpose((1, 2, 0)),
                    suptitle="epoch %d: rmse= %f" % (epoch + 1, rmse),
                )
                writer.add_figure("abu map", fig, epoch + 1, close=True)
                plt.close("all")

        # finish training
        summary_str = (
            f"best rmse = {best_RMSE:.4f}, aad = {best_AAD:.4f}, sad = {best_SAD:.4f}"
        )
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        best_endmember = loadmat(tensorboard_log_dir + "best est_endmember.mat")[
            "est_endmember"
        ]
        sad_endmwise, _ = angle_distance_metric(
            y_Endmember.T, best_endmember.T, verbose=True
        )
        summary_str = f"endmember wise SAD = {sad_endmwise}"
        print(summary_str)
        summary2readme(summary_str, program_log_path + "Readme.txt")

        # close the writer after use
        writer.flush()
        writer.close()

        if best_AAD > aad_guidance and best_SAD > sad_guidance:
            print("delete failed training.")
            shutil.rmtree(program_log_path)

        del net, optimizer, writer

        sema.release()

    ##--------------------------------------------------------------------------------------
    # 5. run in multi-process
    session_num = 1
    processes = []
    sema = mp.Semaphore(value=10)
    num_trials = 5

    for _ in range(num_trials):
        hparams = {
            "EDIP_loss_weight": 1e1,
            "EDIP_angleloss_weight": 100,
            "ADIP_loss_weight": 1e1,
            "ADIP_angleloss_weight": 1e1,
            "BUDIP_loss_weight": 0.1,
            "BUDIP_angleloss_weight": 1e-2,
            "lr": 5e-3,
            "epochs": 12000,
            "down_gamma": 0.9,
            "up_gamma": 0.9,
            "gap": 700,
            "min_gamma": 1e-2,
            "max_gamma": 100,
        }
        sema.acquire()
        run_name = f"run-{session_num}-"
        print("--- Starting trial: %s" % run_name)
        print(hparams)
        p = mp.Process(target=train_test_model, args=(data, run_name, hparams, sema))
        p.start()
        processes.append(p)
        time.sleep(10)
        session_num += 1

    # final loop: close the process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
