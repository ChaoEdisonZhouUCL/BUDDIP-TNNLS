import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


##############################################################################################################
# define functions that write/read data into/from pickle
##############################################################################################################
# define data writer func
def write_data(data, file_path):
    file_writer = open(file_path, "wb")
    pickle.dump(data, file_writer)
    file_writer.close()


# define data reader func
def read_data(file_path):
    file_reader = open(file_path, "rb")
    data = pickle.load(file_reader)
    file_reader.close()
    return data


########################################################################################################################
# plot true and est abundance to filepath
########################################################################################################################
def plot_abundance_map(true_value, est_value, suptitle, **kwargs):
    """

    :param true_value: an array with shape=(nRow,nCol,p), where nRol X nCol is the img size, p is the dimension of latent space/NO_Endms.
                        Describe the true abundance.
    :param est_value: similar to true_value.
    :param kwargs: some possible inputs, like,
                filepath: the filepath where the final visualization figure is saved.
    :return: empty.
    """
    _, _, NO_Endms = true_value.shape
    plt.figure()
    fig, axes = plt.subplots(NO_Endms, 2)
    plt.suptitle(suptitle)

    for p in range(NO_Endms):
        # plot true abundance
        ax = axes[p][0]
        im = ax.imshow(true_value[:, :, p], vmin=0, vmax=1.0, cmap="jet", aspect="auto")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        if p == 0:
            ax.set_title("True")
        plt.colorbar(im, ax=ax)

        # plot est abundance
        ax = axes[p][1]
        im = ax.imshow(est_value[:, :, p], vmin=0, vmax=1.0, cmap="jet", aspect="auto")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        if p == 0:
            ax.set_title("Est")
        plt.colorbar(im, ax=ax)

    fig.set_size_inches(5, 10)
    plt.tight_layout()
    if "filepath" in kwargs.keys():
        plt.savefig(kwargs["filepath"] + "abundance map.png", format="png")
        plt.close("all")
    else:
        return fig


########################################################################################################################
# plot true and est endm_sig to filepath
########################################################################################################################


def plot_Endm(true_value, est_value, suptitle, **kwargs):
    """

    :param true_value: shape=(L,p),the true endm signature, where L is the number of bands, p is the number of endmembers.
    :param est_value: same like true_value.
    :param kwargs: some possible inputs, like,
                    filepath: the filepath where the final visualization figure is saved.
    :return: empty.
    """

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(suptitle)

    ax1.plot(true_value)
    ax1.set_title("True")
    ax2.plot(est_value)
    ax2.set_title("Est")
    if "filepath" in kwargs.keys():
        plt.savefig(kwargs["filepath"] + "Endm Sig.png", format="png")
        plt.close("all")
    else:
        return fig


# {code}
##############################################################################################################
# define project log path creation function
##############################################################################################################
# create project log path
def create_project_log_path(project_path, **kwargs):
    # year_month_day/hour_min/(model_log_dir, model_checkpoint_dir, tensorboard_log_dir)/
    date = datetime.now()
    program_time = project_path + date.strftime("%Y%m%d-%H%M%S")

    Readme_flag = False
    if "Readme" in kwargs.keys():
        Readme_flag = True
    if Readme_flag:
        readme = kwargs.pop("Readme")

    program_log_parent_dir = program_time
    for key, value in kwargs.items():
        program_log_parent_dir = (
            program_log_parent_dir + "_" + key + "_{}".format(value)
        )

    program_log_parent_dir = program_log_parent_dir + "/"
    if not os.path.exists(program_log_parent_dir):
        os.mkdir(program_log_parent_dir)

    # model checkpoint dir
    model_checkpoint_dir = program_log_parent_dir + "model_checkpoint_dir/"
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    # tensorboard_log_dir
    tensorboard_log_dir = program_log_parent_dir + "tensorboard_log_dir/"
    if not os.path.exists(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)

    # model_log_dir
    model_log_dir = program_log_parent_dir + "model_log_dir/"
    if not os.path.exists(model_log_dir):
        os.mkdir(model_log_dir)

    # write exp log
    if Readme_flag:
        with open(program_log_parent_dir + "Readme.txt", "w") as f:
            f.write(readme + "\r\n")
            for key, value in kwargs.items():
                f.write(key + ": {}\r\n".format(value))
            f.write("program log dir: " + program_log_parent_dir + "\r\n")

    return (
        program_log_parent_dir,
        model_checkpoint_dir,
        tensorboard_log_dir,
        model_log_dir,
    )


# write summary to readme.txt
def summary2readme(summary, readme_path):
    with open(readme_path, "a") as fh:
        fh.write(summary)
        fh.write("\r\n")


##############################################################################################################
# from UnDIP
##############################################################################################################
def Eucli_dist(x, y):
    a = np.subtract(x, y)
    return np.dot(a.T, a)


def Endmember_extract(x, p):
    [D, N] = x.shape
    # If no distf given, use Euclidean distance function
    Z1 = np.zeros((1, 1))
    O1 = np.ones((1, 1))
    # Find farthest point
    d = np.zeros((p, N))
    I = np.zeros((p, 1))
    V = np.zeros((1, N))
    ZD = np.zeros((D, 1))
    # if nargin<4
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), ZD)
    # d[0,i]=l1_distance(x[:,i].reshape(D,1),ZD)
    # else
    #     for i=1:N
    #         d(1,i)=distf(x(:,i),zeros(D,1),opt);

    I = np.argmax(d[0, :])

    # if nargin<4
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))
        # d[0,i] = l1_distance(x[:,i].reshape(D,1),x[:,I].reshape(D,1))

    # else
    #     for i=1:N
    #         d(1,i)=distf(x(:,i),x(:,I(1)),opt);
    for v in range(1, p):
        # D=[d[0:v-2,I] ; np.ones((1,v-1)) 0]
        D1 = np.concatenate((d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1)
        D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
        D4 = np.concatenate((D1, D2), axis=0)
        D4 = np.linalg.inv(D4)
        for i in range(N):
            D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
            V[0, i] = np.dot(np.dot(D3.T, D4), D3)

        I = np.append(I, np.argmax(V))
        # if nargin<4
        for i in range(N):
            # d[v,i]=l1_distance(x[:,i].reshape(D,1),x[:,I[v]].reshape(D,1))
            d[v, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I[v]].reshape(D, 1))

        # else
        #     for i=1:N
        #         d(v,i)=distf(x(:,i),x(:,I(v)),opt);
    per = np.argsort(I)
    I = np.sort(I)
    d = d[per, :]
    return I, d


def Endmember_reorder2(A, E1):
    index = []
    _, p = A.shape
    error = np.zeros((1, p))
    for l in range(p):
        for n in range(p):
            error[0, n] = Eucli_dist(A[:, l], E1[:, n])
        b = np.argmin(error)
        index = np.append(index, b)
    index = index.astype(int)
    return index


##############################################################################################################
# convert between np and torch tensor
##############################################################################################################
def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0]


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type="n", var=1.0 / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == "noise":
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1),
        )
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input
