import torch
import numpy as np
import torch.nn as nn

class SparseConv1D(nn.Module):
  
  def __init__(self, C_out, C_in, sk_ind):
    super(SparseConv1D, self).__init__()
    self.C_out = C_out
    self.C_in = C_in 
    self.sk_ind = sk_ind
    self.sk_len = len(sk_ind)
    self.sk_weights = torch.randn(C_out, C_in, self.sk_len, 
                                  dtype=torch.float, requires_grad=True)
    # print('self.sk_weights\n', self.sk_weights)

  def unfold_sparse_1D(self, input_tensor):
    # Find the amount of zero padding needed to make the output the same
    # size as the input.
    input_len = input_tensor.shape[0]
    low_pad = max(0 - min(self.sk_ind), 0)
    high_pad = max(0, max(self.sk_ind))
    input_array = input_tensor.numpy()
    padded_array = np.hstack((input_array, 
                              np.zeros((self.C_in, high_pad)), 
                              np.zeros((self.C_in, low_pad))))
    # print('padded array\n', padded_array)

    # Construct an array of indices that will be used to make the 
    # unfolded array via numpy fancy indexing. 
    # Broadcast to make an array of shape(input_len, sk_len)
    indices = np.arange(input_len)[:, np.newaxis] + self.sk_ind
    # print('indices\n', indices)
  
    # output of array has shape(C_in, input_len, sk_len)
    return torch.tensor(padded_array[np.arange(self.C_in)[:, np.newaxis, np.newaxis], 
                                     indices[np.newaxis, :, :]], 
                                     dtype=torch.float)

  def forward(self, input_tensor):
    input_len = input_tensor.shape[0]
    # Input_array will come in shape (C_in, input_len)
    unfolded = self.unfold_sparse_1D(input_tensor)
    #print('unfolded\n', unfolded)
    #print(self.sk_weights)
    return torch.mm(self.sk_weights.reshape(self.C_out, self.C_in * self.sk_len), 
                    unfolded.reshape(self.C_in * self.sk_len, input_len))

