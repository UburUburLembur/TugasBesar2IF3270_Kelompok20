import numpy as np
import h5py

class Conv2D_Scratch:
    def __init__(self, W, b, padding='same', stride=1):
        self.W, self.b = W, b
        self.padding, self.stride = padding, stride

    def forward(self, x):
        batch, H, W, in_ch = x.shape
        kh, kw, _, out_ch = self.W.shape
        s = self.stride
        if self.padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
            x = np.pad(x, ((0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
        H_out = (x.shape[1] - kh) // s + 1
        W_out = (x.shape[2] - kw) // s + 1
        out = np.zeros((batch, H_out, W_out, out_ch), dtype=np.float32)
        for n in range(batch):
            for i in range(H_out):
                for j in range(W_out):
                    for c in range(out_ch):
                        v0 = i * s
                        h0 = j * s
                        patch = x[n, v0:v0+kh, h0:h0+kw, :]
                        out[n, i, j, c] = np.sum(patch * self.W[..., c]) + self.b[c]
        return out

class Pooling_Scratch:
    def __init__(self, mode='max', pool_size=(2,2), stride=2):
        self.mode = mode
        self.ph, self.pw = pool_size
        self.stride = stride

    def forward(self, x):
        batch, H, W, C = x.shape
        ph, pw, s = self.ph, self.pw, self.stride
        H_out = (H - ph) // s + 1
        W_out = (W - pw) // s + 1
        out = np.zeros((batch, H_out, W_out, C), dtype=x.dtype)
        for n in range(batch):
            for i in range(H_out):
                for j in range(W_out):
                    v0 = i * s
                    h0 = j * s
                    patch = x[n, v0:v0+ph, h0:h0+pw, :]
                    if self.mode == 'max':
                        out[n, i, j, :] = patch.reshape(-1, C).max(axis=0)
                    else:
                        out[n, i, j, :] = patch.reshape(-1, C).mean(axis=0)
        return out

class Flatten_Scratch:
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class GlobalAvgPool2D_Scratch:
    def forward(self, x):
        return x.mean(axis=(1,2))

class Dense_Scratch:
    def __init__(self, W, b):
        self.W, self.b = W, b
    def forward(self, x):
        return x.dot(self.W) + self.b

class ReLU_Scratch:
    def forward(self, x):
        return np.maximum(0, x)

class Softmax_Scratch:
    def forward(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

class CNNFromScratch:
    def __init__(self, h5_filepath, config):
        self.layers = []
        f = h5py.File(h5_filepath, 'r')
        layers_group = f['layers']
        # conv blocks
        for i in range(config['conv_layers']):
            layer_name = 'conv2d' if i == 0 else f'conv2d_{i}'
            vars_group = layers_group[layer_name]['vars']
            W = vars_group['0'][:]
            b = vars_group['1'][:]
            self.layers.append(Conv2D_Scratch(W, b, padding='same', stride=1))
            self.layers.append(ReLU_Scratch())
            if (i+1) % 3 == 0:
                self.layers.append(Pooling_Scratch(mode=config['pooling'], pool_size=(3,3), stride=2))
        # output head
        if config['use_global_avg_pooling']:
            out_name = f'conv2d_{config["conv_layers"]}'
            vars_group = layers_group[out_name]['vars']
            Wf = vars_group['0'][:]
            bf = vars_group['1'][:]
            self.layers.append(Conv2D_Scratch(Wf, bf, padding='valid', stride=1))
            self.layers.append(GlobalAvgPool2D_Scratch())
            self.layers.append(Softmax_Scratch())
        else:
            self.layers.append(Flatten_Scratch())
            # dense layers sequential
            dense_names = [name for name in layers_group if name.startswith('dense')]
            for dn in dense_names:
                vars_group = layers_group[dn]['vars']
                Wd = vars_group['0'][:]
                bd = vars_group['1'][:]
                self.layers.append(Dense_Scratch(Wd, bd))
                self.layers.append(ReLU_Scratch())
            self.layers.append(Softmax_Scratch())
        f.close()

    def forward(self, x):
        out = x.astype(np.float32)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def forward_debug(self, x):
      out = x.astype(np.float32)
      activations = []
      for layer in self.layers:
          out = layer.forward(out)
          activations.append(out)
      return activations

