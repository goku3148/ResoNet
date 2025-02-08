import time
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Module
import matplotlib.pyplot as plt
from  matplotlib import colors
import torch.nn.functional as F
import torch.optim as optim



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets):
        elementwise_loss = self.mse(predictions, targets)
        mask = ~((predictions == 0) & (targets == 0))
        masked_loss = elementwise_loss * mask.float()
        return masked_loss


class CappedReLU(nn.Module):
    def __init__(self):
        super(CappedReLU, self).__init__()

    def forward(self, x):
        return torch.where((x > 0) & (x <= 1), x, torch.zeros_like(x))

class FormMask(Module):
    def __init__(self):
        super(FormMask,self).__init__()
    def forward(self,x,position):
        x_0,y_0,x_1,y_1 = position
        mask = torch.zeros_like(x)
        mask[:,x_0:x_1,y_0:y_1] = x[:,x_0:x_1,y_0:y_1]
        masked_x = mask
        return masked_x

class CWG(Module):#Channel Selection Layer or Channel Wise Gating
    def __init__(self,size,channels,batch_size):
        super(CWG, self).__init__()
        self.batch_size = batch_size
        self.weights = nn.Parameter(torch.Tensor(channels,size))
        self.bias = nn.Parameter(torch.Tensor(size))
        self.relu = nn.ReLU()

        self.weight_init(self.weights,self.bias)

    def weight_init(self,weight:nn.Parameter,bias:nn.Parameter):
        nn.init.xavier_uniform(weight.data)
        nn.init.zeros_(bias.data)

    def forward(self,x):

        if self.batch_size > 1 :
            output = []
            for sample in x:
                diagonal_inte = torch.diagonal(sample.T @ self.weights)
                x = self.relu(diagonal_inte + self.bias)
                output.append(x)
            output = torch.stack(output)
        else :
            diagonal_inte = torch.diagonal(x.T @ self.weights)
            output = self.relu(diagonal_inte + self.bias)

        return output

class GeGate(Module):
    def __init__(self,input_size):
        super(GeGate,self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_size))
        self.bais = nn.Parameter(torch.Tensor(input_size))
        self.fcl = nn.Linear(in_features=input_size,out_features=input_size)

        self.crelu = CappedReLU()
        self.relu = nn.ReLU()
        # self.output_shape = output_shape
        # self.batch_size = batch_size
        self.weight_init(self.weights,self.bais)

    def weight_init(self,weight:nn.Parameter,bias:nn.Parameter):
        nn.init.ones_(weight.data)
        nn.init.zeros_(bias.data)

    def forward(self,x):
        output = []
        for sample in x:
            #sample = self.crelu(self.fcl(sample))
            sample = self.crelu(sample @ torch.diag(self.weights) + self.bais)
            output.append(sample)
        return torch.stack(output)


class Filter(Module):
    #Example input_size : ((30, 30), 2)
    #shape : 30x30 , channels : 2
    def __init__(self, input_shape, batch_size=1, skip_connection:dict={}):
        super(Filter, self).__init__()
        self.batch_size = batch_size
        size, channels = input_shape
        H_in, W_in = size  # should be 30, 30
        self.output_size = H_in * W_in
        self.output_channels = 2 * 6
        assert H_in == W_in == 30, "Input size must be 30x30."

        # Calculate padding for each convolutional layer to keep output size at 30x30
        padding_1x3 = (0,1)  # For kernel (2,2) and stride 2
        padding_3x1 = (1,0)  # For kernel (3,3) and stride 3
        padding_3x3 = (1,1)
        padding_3x5 = (1,2)
        padding_5x3 = (2,1)# For kernel (4,4) and stride 4
        padding_5x5 = (2,2)  # For kernel (5,5) and stride 5
        padding_5x7 = (2, 3)
        padding_7x5 = (3, 2)
        padding_7x7 = (3, 3)
        padding_7x9 = (3, 4)
        padding_9x7 = (2, 2)
        padding_9x9 = (2, 2)

        self.cnn_0 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=(3, 3), stride=(1,1),
                                  padding=padding_3x3)

        cnn_0_p1 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(1, 3), stride=(1,1),
                                  padding=padding_1x3)
        cnn_0_p2 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(3, 1), stride=(1,1),
                                  padding=padding_3x1)
        cnn_0_p3 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(3, 3), stride=(1,1),
                                  padding=padding_3x3)
        cnn_0_p4 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(3, 5), stride=(1,1),
                                  padding=padding_3x5 )
        cnn_0_p5 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(5, 3), stride=(1, 1),
                                  padding=padding_5x3)
        cnn_0_p6 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(5, 5), stride=(1, 1),
                                  padding=padding_5x5)
        cnn_0_p7 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(7, 5), stride=(1, 1),
                             padding=padding_5x5)
        cnn_0_p8 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(5, 7), stride=(1, 1),
                             padding=padding_5x5)
        cnn_0_p9 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(7, 7), stride=(1, 1),
                             padding=padding_5x5)
        cnn_0_p10 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(7, 9), stride=(1, 1),
                             padding=padding_5x5)
        cnn_0_p11 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(9, 7), stride=(1, 1),
                             padding=padding_5x5)
        cnn_0_p12 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=(9, 9), stride=(1, 1),
                             padding=padding_5x5)

        self.C_layer = nn.ModuleList([cnn_0_p1,cnn_0_p2,cnn_0_p3,cnn_0_p4,cnn_0_p5,cnn_0_p6])

        # Define max-pooling layers to reduce the output size by half (to 15x15)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()# This reduces the size to 15x15
        self.flat = nn.Flatten()

    def forward(self,x):
        output = []

        x = self.cnn_0(x)
        x = self.relu(x)

        if self.batch_size > 1:
            for sample in x:
                for i,conv_layer in enumerate(self.C_layer):
                    #con_x = self.pool(self.relu(conv_layer(sample)))
                    con_x = self.relu(conv_layer(sample))
                    mcon_x = con_x if i == 0 else torch.cat((mcon_x, con_x), dim=0)
                output.append(self.flat(mcon_x))
            output = torch.stack(output)
        else :
            for i, conv_layer in enumerate(self.C_layer):
                #con_x = self.pool(conv_layer(x))
                con_x = self.relu(conv_layer(x))
                mcon_x = con_x if i == 0 else torch.cat((mcon_x, con_x), dim=0)
            output = self.flat(mcon_x)
        return output



class RE_Encoder(Module):
    def __init__(self,input_shape,batch_size,generator_grid_shape=(5,5),reso_learning:bool=True):
        super(RE_Encoder,self).__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.reso_learning = reso_learning
        gen_input_size = generator_grid_shape[0] * generator_grid_shape[1]

        self.filter = Filter(input_shape=self.input_shape,batch_size=batch_size)
        self.csl = CWG(size=self.filter.output_size,channels=self.filter.output_channels,batch_size=batch_size)

        self.fcl_0 = nn.Linear(self.filter.output_size,self.filter.output_size*2)
        self.fcl_1 = nn.Linear(self.filter.output_size*2,gen_input_size)
        self.fcl_2 = nn.Linear(gen_input_size,self.filter.output_size)
        self.fcl_3 = nn.Linear(self.filter.output_size,30*30)

        self.flat = nn.Flatten()
        self.en_conv_0 = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=(3,3),padding=(1,1))
        self.en_conv_1 = nn.Conv2d(in_channels=5,out_channels=1,kernel_size=(3,3),padding=(1,1))

        self.gegate = GeGate(30*30)

        self.relu = nn.ReLU()
        self.crelu = CappedReLU()
        self.fmask = FormMask()

    def forward(self,x:Tensor,position):

        filter = self.filter(x)
        csl_x = self.csl(filter)
        x = self.fcl_0(csl_x)
        x = self.fcl_1(torch.round(self.crelu(x)*10)/10)
        gen_input = self.relu(x)
        x = self.fcl_2(gen_input)
        x = self.fcl_3(self.relu(x))
        if self.batch_size > 1:
           x = self.relu(x).reshape(self.batch_size,1,30,30)
        else:
           x = self.relu(x).reshape(1,30,30)
        x = x + csl_x.reshape(1,30,30)
        x = self.en_conv_0(x)
        x = self.en_conv_1(self.crelu(x))
        x = self.fmask(x, position)
        x = self.gegate(self.flat(self.crelu(x)))

        gen_input = gen_input.detach()
        return gen_input,x


class RE_Decoder(Module):
    def __init__(self,input_shape,batch_size,output_shape):
        super(RE_Decoder,self).__init__()
        self.input_shape  = input_shape
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.H_out, self.W_out = output_shape
        output_size = self.H_out * self.W_out

        self.filter = Filter(input_shape=((30,30),1), batch_size=batch_size)
        self.csl = CWG(size=self.filter.output_size, channels=self.filter.output_channels, batch_size=batch_size)

        self.fcl_a = nn.Linear(in_features=self.filter.output_size,out_features=input_shape)

        self.fcl_0 = nn.Linear(in_features=input_shape,out_features=2*input_shape)
        self.fcl_1 = nn.Linear(in_features=2*input_shape,out_features=output_size)


        self.conv_0 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv_1 = nn.Conv2d(in_channels=10,out_channels=1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.gegate = GeGate(30*30)
        self.flat = nn.Flatten()

        self.relu = nn.ReLU()
        self.crelu = CappedReLU()
        self.fmask = FormMask()

    def forward(self,x1,x2,position):

        x2 = self.filter(x2)
        x2 = self.relu(self.csl(self.flat(x2)))
        x2 = self.relu(self.fcl_a(x2))

        x = self.fcl_0(self.relu(x1 + x2))
        x = self.fcl_1(self.relu(x))
        x = self.relu(x)

        if self.batch_size > 1:
            x = x.reshape(self.batch_size,1,self.H_out,self.W_out)
        else:
            x = x.reshape(1,self.H_out,self.W_out)

        x = self.conv_0(x)
        x = self.conv_1(self.crelu(x))
        x = self.fmask(x, position)
        x = self.gegate(self.flat(self.crelu(x)))

        return x


class Y_NET(Module):
    def __init__(self,grid_shape,channels,batch_size,device=None):
        super(Y_NET,self).__init__()
        self.grid_shape = grid_shape
        self.channels = channels
        self.batch_size = batch_size
        self.device = device

        self.encoder = RE_Encoder(input_shape=(grid_shape,channels),batch_size=batch_size,generator_grid_shape=(12,12))
        self.decoder = RE_Decoder(input_shape=12*12,batch_size=batch_size,output_shape=grid_shape)

    def cplot(self, grid, pairs=0, sequence=False, save_path=None, cmap=None, titles=None, figsize=(10, 5)):
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

        plt.show()

    def scaler(self,grid,mode=''):
        if mode == 'up':
            grid = grid + 1
            grid = grid / 10
            return grid
        elif mode == 'down':
            grid = grid * 10
            grid = grid - 1
            grid[grid == -1] = 0
            return grid

    def grid_span(self,grid,mode='up'):
        if mode == 'up':
            s_row, s_col = grid.shape
            d_size = (30, 30)
            m_grid = torch.full(d_size, 0,dtype=grid.dtype)

            r_start = (d_size[0] - s_row) // 2
            c_start = (d_size[1] - s_col) // 2
            m_grid[r_start:r_start + s_row, c_start:c_start + s_col] = grid

            return m_grid.reshape(1,30,30)
        else:
            x_1,y_1,x_2,y_2 = self.get_position(grid)
            l_grid = grid[x_1:x_2,y_1:y_2]
            return l_grid

    def grid_scaler(self,grid,mode='up'):
        if isinstance(grid,list):
            grid = torch.tensor(grid)
        else:
            pass
        if mode == 'up':
            grid = self.scaler(grid,mode='up')
            m_grid = self.grid_span(grid,mode='up')
            return m_grid
        else :
            grid = self.grid_span(grid,mode='down')
            l_grid = self.scaler(grid,mode='down')
            return l_grid

    def data_loader(self,data_pair:dict):
        fit = []
        for i in data_pair['train']:
            in_,out = i['input'],i['output']
            in_,out_ = self.grid_scaler(in_,mode='up'),self.grid_scaler(out,mode='up')
            fit.append([in_,out_])

        shape = torch.tensor(out).shape
        test_out_ = torch.full(shape,0)

        in_,out_ = self.grid_scaler(data_pair['test'][0]['input']),self.grid_scaler(test_out_)
        fit.append([in_,out_])

        return fit

    def en_step(self,ex_output,position):
        gen_input,encoder_output = self.encoder(ex_output,position)
        return encoder_output,gen_input

    def de_step(self,gen_input,input_,position):
        output = self.decoder(gen_input,input_,position)
        return output

    def ssim_loss(self,y_true, y_pred, max_val=1.0):
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

    def mse_loss(self,y_true,y_pred):
        flat = nn.Flatten()
        y_true = flat(y_true)
        mse_loss = nn.MSELoss()
        return mse_loss(y_true,y_pred)

    def masked_mse(self,y_true,y_pred):
        flat = nn.Flatten()
        y_true = flat(y_true)
        mse_loss = MaskedMSELoss()
        return mse_loss(y_true,y_pred)

    def discretize_output(self,output, levels=11):
        discrete_values = torch.linspace(0, 1, levels).to(output.device)  # e.g., [0.0, 0.1, ..., 1.0]

        # Find the nearest discrete value for each element in output
        output_discretized = torch.zeros_like(output)
        for i, val in enumerate(discrete_values):
            output_discretized += torch.abs(output - val).argmin(dim=-1, keepdim=True).float() * val
        return output_discretized

    def get_position(self,matrix):
        valid_indices = torch.nonzero(matrix != 0, as_tuple=False)
        position = (valid_indices[0][0],valid_indices[0][1],valid_indices[-1][0]+1,valid_indices[-1][1]+1)
        return position

    def test_fit(self, arc_problem, epochs=100, activation=50, en_lr=0.009, de_lr=0.0005, outputs=4):

        data_pairs =  self.data_loader(arc_problem)

        optimizer_en = optim.Adam(self.encoder.parameters(), lr=en_lr)
        optimizer_de = optim.Adam(self.decoder.parameters(), lr=de_lr)

        iterations = len(data_pairs)
        pred = []
        ite = 0
        loss_en_,loss_de_ = [],[]
        in_out = []
        act = int((epochs - 100) / outputs)
        while ite < epochs:
            for i in range(iterations - 1):
                y_in = data_pairs[i][0]
                y_out = data_pairs[i][1]
                pos_y_out = self.get_position(y_out[0])
                pos_y_in = self.get_position(y_in[0])


                pred_in, gen_input = self.en_step(ex_output=y_out,position=pos_y_in)
                loss_en = self.masked_mse(y_in, pred_in)
                en_gt = torch.ones_like(loss_en)

                loss_en.backward(gradient=en_gt)
                optimizer_en.step()
                optimizer_en.zero_grad()

                pred_out = self.de_step(gen_input=gen_input,input_=y_in,position=pos_y_out)
                loss_de = self.masked_mse(y_out, pred_out)
                de_gt = torch.ones_like(loss_de)

                loss_de.backward(gradient=de_gt)
                optimizer_de.step()
                optimizer_de.zero_grad()

            if ite > activation:

                y_in = data_pairs[-1][0]
                y_out = data_pairs[-1][1]

                pred_in, gen_input = self.en_step(ex_output=y_out, position=pos_y_in)
                loss_en = self.masked_mse(y_in, pred_in)
                en_gt = torch.ones_like(loss_en)

                loss_en.backward(gradient=en_gt)
                optimizer_en.step()
                optimizer_en.zero_grad()

                pred_out = self.de_step(gen_input=gen_input, input_=y_in, position=pos_y_out)
                loss_de = self.masked_mse(y_out, pred_out)
                de_gt = torch.ones_like(loss_de)

                loss_de.backward(gradient=de_gt)
                optimizer_de.step()
                optimizer_de.zero_grad()
                data_pairs[-1][1] = torch.round((pred_out.detach().reshape(1,30,30))*10)/10

            ite += 1
            loss_en_.append(float(loss_de.mean())),loss_de_.append(float(loss_de.mean()))

            print(ite, act + 100 == ite, act)
            
            if act + 200 == ite:
                try :
                    scaled_input, scaled_output = self.grid_scaler(data_pairs[-1][0][0],mode='down'), self.grid_scaler(data_pairs[-1][1][0], mode='down')
                    in_out.append([scaled_input, scaled_output])
                except Exception as e:
                    print(e)
                act += act
        in_out.append([scaled_input, scaled_output])

        return in_out,loss_en_,loss_de_

