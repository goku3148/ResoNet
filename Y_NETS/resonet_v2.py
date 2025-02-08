import math
import random
import time
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Module
import matplotlib.pyplot as plt
from matplotlib import colors
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.optim.lr_scheduler import LambdaLR


class MaskedMSELoss_element_wise(nn.Module):
    def __init__(self,device='cuda'):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.device = device

    def forward(self, predictions, targets):
        elementwise_loss = self.mse(predictions, targets).to(self.device)
        mask = ~((predictions == 0) & (targets == 0)).to(self.device)
        masked_loss = elementwise_loss * mask.float()
        return masked_loss

class MaskedMSELoss(nn.Module):
        def __init__(self, device='cuda'):
            super(MaskedMSELoss, self).__init__()
            self.mse = nn.MSELoss(reduction='none')
            self.device = device

        def forward(self, predictions, targets):
            elementwise_loss = self.mse(predictions, targets).to(self.device)
            mask = ~((predictions == 0) & (targets == 0)).to(self.device)
            masked_loss = elementwise_loss * mask.float()
            return masked_loss.mean()


class CappedReLU(nn.Module):
    def __init__(self,device):
        super(CappedReLU, self).__init__()
        self.device = device

    def forward(self, x):
        return torch.where((x > 0) & (x <= 1), x, torch.zeros_like(x)).to(self.device)


class FormMask(Module):
    def __init__(self):
        super(FormMask, self).__init__()

    def forward(self, x, position):
        x_0, y_0, x_1, y_1 = position
        mask = torch.zeros_like(x)
        mask[:,:, x_0:x_1, y_0:y_1] = x[:,:, x_0:x_1, y_0:y_1]
        masked_x = mask
        return masked_x


class CWG(Module):  # Channel Selection Layer or Channel Wise Gating
    def __init__(self, size, channels, batch_size, node_activation=False, device='cpu'):
        super(CWG, self).__init__()
        self.batch_size = batch_size
        self.weights = nn.Parameter(torch.Tensor(channels, size)).to(device)
        self.bias = nn.Parameter(torch.Tensor(size)).to(device)
        self.crelu = CappedReLU(device=device)
        self.node_activation = node_activation

        self.weight_init(self.weights, self.bias)

    def weight_init(self, weight: nn.Parameter, bias: nn.Parameter):
        nn.init.xavier_uniform(weight.data)
        nn.init.zeros_(bias.data)

    def forward(self, x):

        output = []
        for sample in x:
            diagonal_inte = torch.diagonal(sample.T @ self.weights)
            x = diagonal_inte + self.bias
            output.append(x)
        output = torch.stack(output)
        output = self.crelu(output) if self.node_activation else output
        return output


class GeGate(Module):
    def __init__(self, input_size, device):
        super(GeGate, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_size)).to(device)
        self.bais = nn.Parameter(torch.Tensor(input_size)).to(device)
        self.crelu = CappedReLU(device=device)
        self.relu = nn.ReLU()
        self.weight_init(self.weights, self.bais)

    def weight_init(self, weight: nn.Parameter, bias: nn.Parameter):
        nn.init.ones_(weight.data)
        nn.init.zeros_(bias.data)

    def forward(self, x):
        output = []
        for sample in x:
            # sample = self.crelu(self.fcl(sample))
            sample = self.crelu(sample @ torch.diag(self.weights) + self.bais)
            output.append(sample)
        return torch.stack(output)


class IDEA_PROJ(Module):
    # Example input_size : ((30, 30), 2)
    # shape : 30x30 , channels : 2
    def __init__(self, input_shape, batch_size=1,node_activation=False, device='cpu'):
        super(IDEA_PROJ, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.node_activation = node_activation
        channels, H_in, W_in = input_shape
        self.output_size = H_in * W_in
        self.input_channel = channels
        self.output_channels = 1
        self.t_output_channels = 12 * self.output_channels

        # Calculate padding for each convolutional layer to keep output size at 30x30
        padding_1x3 = (0, 1)  # For kernel (2,2) and stride 2
        padding_3x1 = (1, 0)  # For kernel (3,3) and stride 3
        padding_3x3 = (1, 1)
        padding_3x5 = (1, 2)
        padding_5x3 = (2, 1)  # For kernel (4,4) and stride 4
        padding_5x5 = (2, 2)  # For kernel (5,5) and stride 5
        padding_5x7 = (2, 3)
        padding_7x5 = (3, 2)
        padding_7x7 = (3, 3)
        padding_7x9 = (3, 4)
        padding_9x7 = (4, 3)
        padding_9x9 = (4, 4)

        cnn_0_p1 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(1, 3), stride=(1, 1),
                             padding=padding_1x3).to(self.device)
        cnn_0_p2 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(3, 1), stride=(1, 1),
                             padding=padding_3x1).to(self.device)
        cnn_0_p3 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(3, 3), stride=(1, 1),
                             padding=padding_3x3).to(self.device)
        cnn_0_p4 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(3, 5), stride=(1, 1),
                             padding=padding_3x5).to(self.device)
        cnn_0_p5 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(5, 3), stride=(1, 1),
                             padding=padding_5x3).to(self.device)
        cnn_0_p6 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(5, 5), stride=(1, 1),
                             padding=padding_5x5).to(self.device)
        cnn_0_p7 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(7, 5), stride=(1, 1),
                             padding=padding_7x5).to(self.device)
        cnn_0_p8 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(5, 7), stride=(1, 1),
                             padding=padding_5x7).to(self.device)
        cnn_0_p9 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(7, 7), stride=(1, 1),
                             padding=padding_7x7).to(self.device)
        cnn_0_p10 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(7, 9), stride=(1, 1),
                              padding=padding_7x9).to(self.device)
        cnn_0_p11 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(9, 7), stride=(1, 1),
                              padding=padding_9x7).to(self.device)
        cnn_0_p12 = nn.Conv2d(in_channels=channels, out_channels=self.output_channels, kernel_size=(9, 9), stride=(1, 1),
                              padding=padding_9x9).to(self.device)

        self.C_layers = nn.ModuleList([cnn_0_p1, cnn_0_p2, cnn_0_p3, cnn_0_p4, cnn_0_p5, cnn_0_p6, cnn_0_p7, cnn_0_p8, cnn_0_p9, cnn_0_p10, cnn_0_p11, cnn_0_p12])

        # Define max-pooling layers to reduce the output size by half (to 15x15)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()

        self.csl = CWG(H_in*W_in,self.t_output_channels,batch_size,device=self.device,node_activation=False)

    def forward(self, x):


        batch,ch,x_h,x_w = x.shape
        conv_i = []
        for i, conv_layer in enumerate(self.C_layers):
            con_x = conv_layer(x)
            conv_i.append(con_x)

        conv_i = torch.stack(conv_i)
        conv_i = conv_i.permute(1, 0, 2, 3, 4).to(self.device)
        conv_i = conv_i.reshape(batch, -1, x_h, x_w)
        conv_i = conv_i.view(batch, conv_i.size(1), -1)

        conv_i = self.relu(conv_i)
        output = self.csl(conv_i)

        output = output.reshape(batch,self.input_channel, x_h, x_w)

        return output

class RE_CONS(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, output_padding=0, node_activation=False, device='cpu',proj='conv'):
        super(RE_CONS, self).__init__()
        self.decode = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        ).to(device)
        self.proj = proj
        self.node_activation = node_activation
        self.activation = nn.ReLU()  # Use ReLU as activation (or customize)

    def forward(self, x):

        x = self.decode(x)
        x = self.activation(x) if self.node_activation else x

        return x

class RE_Encoder(Module):
    def __init__(self, input_shape=(1,30,30), batch_size=1, gen_size=50, reso_learning: bool = True,device='cpu'):
        super(RE_Encoder, self).__init__()
        self.input_shape = input_shape
        self.channels,H_in,W_in = input_shape
        self.shape = (H_in,W_in)
        self.batch_size = batch_size
        self.reso_learning = reso_learning
        gen_input_size = gen_size
        self.linear_feature = H_in * W_in
        linear_construction = 400

        self.idea_proj_0 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=True)
        self.idea_proj_1 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=False)
        self.idea_proj_2 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=True)
        self.idea_proj_3 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=False)

        self.fcl_d_0 = nn.Linear(in_features=self.linear_feature,out_features=self.linear_feature).to(device)
        self.fcl_d_1 = nn.Linear(in_features=self.linear_feature, out_features=gen_input_size).to(device)
        self.fcl_u_0 = nn.Linear(in_features=gen_input_size, out_features=self.linear_feature).to(device)
        self.fcl_u_1 = nn.Linear(in_features=self.linear_feature, out_features=self.linear_feature).to(device)

        self.recon_1 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=True)
        self.recon_2 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=False)
        self.recon_3 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=True)
        self.gegate = GeGate(self.linear_feature,device=device)

        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.crelu = CappedReLU(device)
        self.fmask = FormMask()

    def forward(self, x: Tensor):

        batch,channel,h_x,w_x = x.shape
        x_0 = x
        x = self.idea_proj_0(x)
        x = self.idea_proj_1(x)
        x_0 = self.idea_proj_2(self.crelu(x + x_0))
        x = self.idea_proj_3(x_0)
        x = self.crelu(x + x_0)
        x = self.flat(x)

        x = self.relu(self.fcl_d_0(x))
        gen = self.relu(self.fcl_d_1(x))
        x = self.relu(self.fcl_u_0(gen))
        x = self.relu(self.fcl_u_1(x))
        
        x_0 = x.reshape(batch,1,self.shape[0],self.shape[1])
        x = self.recon_1(x_0)
        x = self.relu(self.recon_2(x + x_0))
        x = self.recon_3(x)
        x = self.flat(x)
        output = self.gegate(x)

        return gen, output


class RE_Decoder(Module):
    def __init__(self, input_shape, gen_size, batch_size, device):
        super(RE_Decoder, self).__init__()

        self.device = device
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.channels,self.H_out, self.W_out = input_shape
        output_size = self.H_out * self.W_out

        self.idea_proj_0 = IDEA_PROJ(input_shape=input_shape,batch_size=batch_size,device=device,node_activation=True)
        self.idea_proj_1 = IDEA_PROJ(input_shape=input_shape,batch_size=batch_size,device=device,node_activation=True)
        self.idea_proj_2 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device,node_activation=True)

        self.fcl_0 = nn.Linear(in_features=gen_size, out_features=output_size).to(device)
        self.fcl_1 = nn.Linear(in_features=output_size, out_features=output_size).to(device)
        self.fcl_2 = nn.Linear(in_features=output_size, out_features=output_size).to(device)

        self.recon_1 = IDEA_PROJ(input_shape=input_shape,batch_size=batch_size,device=device,node_activation=True)
        self.recon_2 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=False)
        self.recon_3 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=True)
        self.recon_4 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=False)
        self.recon_5 = IDEA_PROJ(input_shape=input_shape, batch_size=batch_size, device=device, node_activation=False)

        self.gegate = GeGate(output_size,device)
        self.flat = nn.Flatten()

        self.relu = nn.ReLU()
        self.crelu = CappedReLU(device)
        self.fmask = FormMask()

    def forward(self, x_en, x, position):

        batch, channel, h_x, w_x = x.shape
        x_in = self.idea_proj_0(x)
        x = self.idea_proj_1(x_in)
        x = self.idea_proj_2(x)
        x = self.flat(x)

        x_en = self.relu(self.fcl_0(x_en))
        x = self.relu(self.fcl_1(x_en + x_en))
        x = self.relu(self.fcl_2(x))

        x_0 = x.reshape(batch,self.channels,self.H_out,self.W_out)

        x = self.recon_1(x_0 + x_in)
        x_1 = self.relu(self.recon_2(x_0 + x))
        x = self.recon_3(x_0)
        x_2 = self.relu(self.recon_4(x_0 + x))

        x = self.relu(self.recon_5(self.relu(x_1 + x_2)))

        #print(x)
        #x = self.fmask(x,position)
        x = self.flat(x)
        x = self.gegate(x)
        return x


class Y_NET(Module):
    def __init__(self, grid_shape, channels, batch_size, neck_size=100, device=None):
        super(Y_NET, self).__init__()
        self.grid_shape = grid_shape
        self.channels,self.H_X,self.W_X = grid_shape
        self.batch_size = batch_size
        self.device = device
        self.neck_size = neck_size

        self.encoder = RE_Encoder(input_shape=grid_shape, batch_size=batch_size,device=device,gen_size=self.neck_size)
        self.decoder = RE_Decoder(input_shape=grid_shape, batch_size=batch_size,device=device,gen_size=neck_size)

    def cplot(self, grid, pairs=0, sequence=False, save_path=None, cmap=None, titles=None, figsize=(10, 5), show:bool=True):
        """
        Visualizes the grid, pairs of grids, or a sequence of images.

        Args:
            grid (array-like): The grid or list of grids to visualize.
            pairs (int, optional): Number of grid pairs to plot. Default is 0 (single grid mode).
            sequence (bool, optional): Whether to treat the input as a sequence of images. Default is False.
            save_path (str, optional): Path to save the plot as an image. Default is None.
            cmap (str or Colormap, optional): Colormap to use for visualization. Default is a custom discrete colormap.
            titles (list of str, optional): Titles for subplots if sequence=True or pairs > 0.
            figsize (tuple, optional): Figure size for the plot. Default is (10, 5).
        """
        if cmap is None:
            _cmap = colors.ListedColormap(
                ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                 '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
            )
        else:
            _cmap = plt.get_cmap(cmap)

        norm = colors.Normalize(vmin=0, vmax=9)

        if sequence:
            num_images = len(grid)
            fig, axs = plt.subplots(1, num_images, figsize=figsize)
            if num_images == 1:
                axs = [axs]

            for i, img in enumerate(grid):
                axs[i].imshow(img, norm=norm, cmap=_cmap)
                axs[i].axis('off')
                if titles and len(titles) > i:
                    axs[i].set_title(titles[i])

            plt.tight_layout()

        elif pairs > 0:
            fig, axs = plt.subplots(nrows=2, ncols=pairs, figsize=figsize, squeeze=False)
            for i in range(pairs):
                axs[0][i].imshow(grid[i][0], norm=norm, cmap=_cmap)
                axs[1][i].imshow(grid[i][1], norm=norm, cmap=_cmap)

                if titles and len(titles) > i:
                    axs[0][i].set_title(titles[i])

            for ax_row in axs:
                for ax in ax_row:
                    ax.axis('off')

            plt.tight_layout()

        else:
            plt.figure(figsize=figsize)
            plt.imshow(grid, norm=norm, cmap=_cmap)
            plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()

    def progress_bar(self,iterable, total=None, prefix='', suffix='', length=50, fill='█', print_end='\r'):
        """
        A terminal progress bar function.

        Args:
            iterable: The iterable to loop over.
            total (int): Total iterations (if None, will be inferred from len(iterable)).
            prefix (str): Prefix string.
            suffix (str): Suffix string.
            length (int): Length of the progress bar.
            fill (str): Character to show the filled portion of the bar.
            print_end (str): End character (e.g., '\n', '\r').

        Yields:
            Items from the iterable.
        """
        if total is None:
            total = len(iterable)

        def print_bar(iteration):
            percent = ("{0:.1f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + '-' * (length - filled_length)
            sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
            sys.stdout.flush()
            if iteration == total:
                sys.stdout.write(print_end)

        for i, item in enumerate(iterable, start=1):
            yield item
            print_bar(i)
        print_bar(total)

    def scaler(self, grid, mode=''):
        if mode == 'up':
            grid = grid + 1
            grid = grid / 10
            return grid
        elif mode == 'down':
            grid = grid * 10
            grid = grid - 1
            grid[grid == -1] = 0
            return grid

    def grid_span(self, grid, mode='up'):
        if mode == 'up':
            s_row, s_col = grid.shape
            d_size = (30, 30)
            m_grid = torch.full(d_size, 0, dtype=grid.dtype)

            r_start = (d_size[0] - s_row) // 2
            c_start = (d_size[1] - s_col) // 2
            m_grid[r_start:r_start + s_row, c_start:c_start + s_col] = grid

            return m_grid.reshape(1, 30, 30)
        else:
            x_1, y_1, x_2, y_2 = self.get_position(grid)
            l_grid = grid[x_1:x_2, y_1:y_2]
            return l_grid

    def grid_scaler(self, grid, mode='up'):
        if isinstance(grid, list):
            grid = torch.tensor(grid)
        else:
            pass
        if mode == 'up':
            grid = self.scaler(grid, mode='up')
            m_grid = self.grid_span(grid, mode='up')
            return m_grid
        else:
            grid = self.grid_span(grid, mode='down')
            l_grid = self.scaler(grid, mode='down')
            return l_grid

    def data_loader(self, data_pairs: dict, loader: str='test', batch_size: int=16):
        if loader == 'test':
            fit = []
            data_pair = data_pairs
            for i in data_pair['train']:
                in_, out = i['input'], i['output']
                in_, out_ = self.grid_scaler(in_, mode='up'), self.grid_scaler(out, mode='up')
                fit.append([in_, out_])

            shape = torch.tensor(out).shape
            test_out_ = torch.full((30,30), 0)

            in_, out_ = self.grid_scaler(data_pair['test'][0]['input']), self.grid_scaler(test_out_)
            fit.append([in_, out_])

            return fit

        elif loader == 'train':
            data_pairs = list(data_pairs.values())
            samples = []
            for i in data_pairs:
                for j in i['train']:
                    in_, out = j['input'], j['output']
                    in_, out_ = self.grid_scaler(in_, mode='up').reshape(1,30,30), self.grid_scaler(out, mode='up').reshape(1,30,30)
                    samples.append([in_,out_])

            size = len(samples)
            random.shuffle(samples)

            n_batches = int(size / batch_size)
            in_,out_=[],[]

            for i in range(n_batches):
               batch_ = samples[batch_size*i : batch_size*(i + 1)]
               in_b,out_b = zip(*batch_)
               in_.append(torch.stack(list(in_b)))
               out_.append(torch.stack(list(out_b)))

        elif loader == 'eval_train':
            data_pair = data_pairs
            in__,out__ = [],[]
            for i in data_pair['train']:
                in_, out_ = i['input'], i['output']
                in_, out_ = self.grid_scaler(in_, mode='up'), self.grid_scaler(out_, mode='up')
                in__.append(in_),out__.append(out_)
            train = [torch.stack(in__),torch.stack(out__)]
            test_in = self.grid_scaler(data_pair['test'][0]['input']).unsqueeze(1)
            test_out = self.grid_scaler(torch.full((30, 30), 0),mode='up').unsqueeze(1)
            test = [test_in,test_out]
            return train,test

    def en_step(self, ex_output, position):
        gen_input, encoder_output = self.encoder(ex_output)
        return encoder_output, gen_input

    def de_step(self, gen_input, input_, position):
        output = self.decoder(gen_input, input_, position)
        return output

    def ssim_loss(self, y_true, y_pred, max_val=1.0):
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2

        # Calculate means
        mu_x = F.avg_pool2d(y_true, 3, 1)
        mu_y = F.avg_pool2d(y_pred, 3, 1)

        # Calculate variances and covariance
        sigma_x = F.avg_pool2d(y_true ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y_pred ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(y_true * y_pred, 3, 1) - mu_x * mu_y

        # Calculate SSIM
        ssim_index = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        # Convert SSIM to SSIM loss
        return 1 - ssim_index.mean()

    def mse_loss(self, y_true, y_pred):
        flat = nn.Flatten()
        y_true = flat(y_true)
        mse_loss = nn.MSELoss()
        return mse_loss(y_true, y_pred)

    def masked_mse(self, y_true, y_pred):
        flat = nn.Flatten()
        y_true = flat(y_true)
        mse_loss = MaskedMSELoss()
        return mse_loss(y_true, y_pred)

    def discretize_output(self, output, levels=11):
        discrete_values = torch.linspace(0, 1, levels).to(output.device)  # e.g., [0.0, 0.1, ..., 1.0]

        # Find the nearest discrete value for each element in output
        output_discretized = torch.zeros_like(output)
        for i, val in enumerate(discrete_values):
            output_discretized += torch.abs(output - val).argmin(dim=-1, keepdim=True).float() * val
        return output_discretized

    def get_position(self, matrix):
        valid_indices = torch.nonzero(matrix != 0, as_tuple=False)
        position = (valid_indices[0][0], valid_indices[0][1], valid_indices[-1][0] + 1, valid_indices[-1][1] + 1)
        return position

    def train_fit(self, dataset, batch_size=1, epochs=500, en_lr=0.009, de_lr=0.005, save_path=''):

        optimizer_en = optim.Adam(self.encoder.parameters(), lr=en_lr)
        optimizer_de = optim.Adam(self.decoder.parameters(), lr=de_lr)
        loss_en_, loss_de_ = [], []

        size = len(dataset)
        iterations = range(epochs)
        for epo in self.progress_bar(iterations,total=epochs,prefix='Progress'):
            loss_en_step, loss_de_step = 0,0
            for ite in range(size):
                y_in, y_out = dataset[0][ite].to(self.device), dataset[1][ite].to(self.device)
                pred_in, gen_input = self.en_step(ex_output=y_out, position=(0,0,0,0))

                loss_en = self.masked_mse(y_in, pred_in)
                loss_en.backward()
                optimizer_en.step()
                optimizer_en.zero_grad()

                pred_out = self.de_step(gen_input=gen_input.detach(), input_=y_in, position=(0,0,0,0))
                loss_de = self.masked_mse(y_out, pred_out)

                loss_de.backward()
                optimizer_de.step()
                optimizer_de.zero_grad()

                loss_en_step += loss_en
                loss_de_step += loss_de
            loss_en_.append(float(loss_en_step.to('cpu')/size))
            loss_de_.append(float(loss_de_step.to('cpu')/size))

        return loss_en_,loss_de_

    def test_fit(self, arc_problem, epochs=100, activation=50, en_lr=0.009, de_lr=0.005, outputs=4):

        data_pairs = self.data_loader(arc_problem)

        optimizer_en = optim.Adam(self.encoder.parameters(), lr=en_lr)
        optimizer_de = optim.Adam(self.decoder.parameters(), lr=de_lr)

        iterations = len(data_pairs)
        pred = []
        ite = 0
        loss_en_, loss_de_ = [], []
        in_out = []
        act = int((epochs - 100) / outputs)
        iterable = range(epochs)
        for ite in self.progress_bar(iterable,total=epochs,prefix="Progress"):
            for i in range(iterations - 1):
                y_in = data_pairs[i][0].to(self.device)
                y_in = y_in.reshape(self.batch_size, self.channels, self.H_X, self.W_X)
                y_out = data_pairs[i][1].to(self.device)
                y_out = y_out.reshape(self.batch_size,self.channels,self.H_X,self.W_X)

                pos_y_out = self.get_position(y_out[0][0])
                pos_y_in = self.get_position(y_in[0][0])

                pred_in, gen_input = self.en_step(ex_output=y_out, position=pos_y_in)
                loss_en = self.masked_mse(y_in, pred_in)
                #en_gt = torch.ones_like(loss_en)

                loss_en.backward()#gradient=en_gt)
                optimizer_en.step()
                optimizer_en.zero_grad()

                pred_out = self.de_step(gen_input=gen_input.detach(), input_=y_in, position=pos_y_out)
                loss_de = self.masked_mse(y_out, pred_out)
                #de_gt = torch.ones_like(loss_de)

                loss_de.backward()#gradient=de_gt)
                optimizer_de.step()
                optimizer_de.zero_grad()
                data_pairs[-1][1] = torch.round((pred_out.detach().reshape(1, 1, 30, 30)) * 10) / 10

            if ite > activation:
                y_in = data_pairs[-1][0].to(self.device)
                y_in = y_in.reshape(self.batch_size, self.channels, self.H_X, self.W_X)
                y_out = data_pairs[-1][1].to(self.device)
                y_out = y_out.reshape(self.batch_size, self.channels, self.H_X, self.W_X)

                pred_in, gen_input = self.en_step(ex_output=y_out, position=pos_y_in)
                loss_en = self.masked_mse(y_in, pred_in)
                en_gt = torch.ones_like(loss_en)

                loss_en.backward(gradient=en_gt)
                optimizer_en.step()
                optimizer_en.zero_grad()

                pred_out = self.de_step(gen_input=gen_input.detach(), input_=y_in, position=pos_y_out)
                loss_de = self.masked_mse(y_out, pred_out)
                de_gt = torch.ones_like(loss_de)

                #loss_de.backward(gradient=de_gt)
                #optimizer_de.step()
                #optimizer_de.zero_grad()
                data_pairs[-1][1] = torch.round((pred_out.detach().reshape(1 ,1, 30, 30)) * 10) / 10



            loss_en_.append(float(loss_en.to('cpu').mean())), loss_de_.append(float(loss_de.to('cpu').mean()))

            outputs = int(epochs - activation / outputs)

            if ite > activation and ite % outputs == 0:
                try:
                    scaled_input, scaled_output = self.grid_scaler(data_pairs[-1][0][0], mode='down'), self.grid_scaler(
                        data_pairs[-1][1][0].reshape(30,30).to('cpu'), mode='down')
                    in_out.append([scaled_input, scaled_output])
                except Exception as e:
                    print(e)
                act += act
        try:
            scaled_input, scaled_output = self.grid_scaler(data_pairs[-1][0][0], mode='down'), self.grid_scaler(
                data_pairs[-1][1][0].reshape(30, 30).to('cpu'), mode='down')
            in_out.append([scaled_input, scaled_output])
        except Exception as e:
            print(e)

        return in_out, loss_en_, loss_de_

    def test_eval(self, arc_problem, epochs=100, activation=50, en_lr=0.009, de_lr=0.005, noutput=5):
        train,test = self.data_loader(arc_problem,loader='eval_train')

        optimizer_en = optim.Adam(self.encoder.parameters(), lr=en_lr)
        optimizer_de = optim.Adam(self.decoder.parameters(), lr=de_lr)
        loss_en_, loss_de_ = [], []

        iterations = range(epochs)

        prediction = []

        for epo in self.progress_bar(iterations,total=epochs,prefix='Progress'):

            y_in, y_out = train[0].to(self.device), train[1].to(self.device)
            pred_in, gen_input = self.en_step(ex_output=y_out, position=(0, 0, 0, 0))

            loss_en = self.masked_mse(y_in, pred_in)
            loss_en.backward()
            optimizer_en.step()
            optimizer_en.zero_grad()

            pred_out = self.de_step(gen_input=gen_input.detach(), input_=y_in, position=(0, 0, 0, 0))
            loss_de = self.masked_mse(y_out, pred_out)

            loss_de.backward()
            optimizer_de.step()
            optimizer_de.zero_grad()



            if epo > activation:
                y_in, y_out = test[0].to(self.device), test[1].to(self.device)
                pred_in, gen_input = self.en_step(ex_output=y_out, position=(0, 0, 0, 0))

                loss_en = self.masked_mse(y_in, pred_in)
                loss_en.backward()
                optimizer_en.step()
                optimizer_en.zero_grad()

                pred_out = self.de_step(gen_input=gen_input.detach(), input_=y_in, position=(0, 0, 0, 0))
                loss_de = self.masked_mse(y_out, pred_out)

                #loss_de.backward()
                #optimizer_de.step()
                #optimizer_de.zero_grad()
                m_pred = torch.round((pred_out.detach()).view(1,1,30,30) * 10) / 10
                test[1] = m_pred
                prediction.append(m_pred)

            loss_en_.append(float(loss_en.to('cpu'))),loss_de_.append(float(loss_de.to('cpu')))

        prediction.reverse()
        scat = int((epochs - activation)/noutput)
        prediction = [prediction[i*scat] for i in range(noutput)]
        prediction = torch.stack(prediction).view(noutput,30,30)
        prediction = prediction.to('cpu')

        return prediction,loss_en_,loss_de_



pair = torch.rand(size=(16,1,30,30)).to('cuda')
shape = pair.shape
batch = 16

encoder = RE_Encoder(batch_size=batch,device='cuda')
gen,en = encoder(pair)
decoder = RE_Decoder(input_shape=encoder.input_shape,gen_size=50,batch_size=batch,device='cuda')
output = decoder(gen,pair,(0,0,0,0))



