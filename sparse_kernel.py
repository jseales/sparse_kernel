import torch
import numpy as np
import torch.nn as nn

class SparseConv1D(nn.Module):
  
  def __init__(self, sk_ind, in_channels=1, out_channels=1):
    super(SparseConv1D, self).__init__()
    self.out_channels = out_channels
    self.in_channels = in_channels 
    self.sk_ind = sk_ind
    self.sk_len = len(sk_ind)
    self.sk_weights = torch.randn(out_channels, in_channels, self.sk_len, 
                                  dtype=torch.float, requires_grad=True)
    #print('self.sk_weights\n', self.sk_weights)


  def unfold_sparse_1D(self, input_tensor):
    # Find the amount of zero padding needed to make the output the same
    # size as the input.
    print('input_tensor.shape', input_tensor.shape)
    low_pad = int(max(0 - min(self.sk_ind), 0))
    high_pad = int(max(0, max(self.sk_ind)))
    input_array = input_tensor.cpu().detach().numpy()
    padded_array = np.hstack((input_array, 
                              np.zeros((self.in_channels, high_pad)), 
                              np.zeros((self.in_channels, low_pad))))
    print('padded array\n', padded_array)

    # Construct an array of indices that will be used to make the 
    # unfolded array via numpy fancy indexing. 
    # Broadcast to make an array of shape(sk_len, input_len)
    indices = sk_ind[:, np.newaxis] + np.arange(self.input_len)
    print('indices\n', indices)
    # output of array has shape(in_channels, sk_len, input_len)
    return torch.tensor(padded_array[np.arange(self.in_channels)[:, np.newaxis, np.newaxis], 
                                     indices[np.newaxis, :, :]], 
                                     dtype=torch.float)

  def forward(self, input_tensor):
    self.input_len = input_tensor.shape[1]
    # Input_array will come in shape (in_channels, input_len)
    unfolded = self.unfold_sparse_1D(input_tensor)
    print('unfolded\n', unfolded)
    #print(self.sk_weights)
    return torch.mm(self.sk_weights.reshape(self.out_channels, self.in_channels * self.sk_len), 
                    unfolded.reshape(self.in_channels * self.sk_len, self.input_len))
