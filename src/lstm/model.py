import numpy as np
import re
import keras
import h5py
import copy

def sigmoid(x):
  s = 1/(1 + np.exp(-(x)))
  return s

def softmax(x):
  if len(x.shape) == 1:
    x = np.array([x])
  elif len(x.shape) != 2:
    raise ValueError(f"Input harus berupa array 1D atau 2D (sample, classes), didapat {len(x.shape)}D")

  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x, axis=1)

  p = np.array([exp_x[i,:]/sum_exp_x[i] for i in range(len(x))])
  return p

def relu(x):
  return np.maximum(0, x)

def batch_array(x, batch_size):
  for i in range(0, len(x), batch_size):
    yield x[i:i+batch_size]
    
class sequential:
  def __init__(self):
    self.seq = []
    self.layer_names = []
    self.weights = []

  def get_weights(self, fname):   # baca file .weights.h5, ambil bobot masing-masing layer
    weights = {}
    names = []

    with h5py.File(fname, "r") as f:
      layers = f["layers"]
      name = layers.keys()

      for n in name:
        names.append(n)
        w = layers[n]["vars"]

        if "dense" == re.sub(r'[^a-zA-Z\s]', '', n):
          w = layers[n]["vars"]
          weight_layer = [w["0"][:], w["1"][:]]

        elif "lstm" == re.sub(r'[^a-zA-Z\s]', '', n):
          w = layers[n]["cell"]["vars"]
          units = w["1"].shape[0]
          weight_layer = [w["0"][:], w["1"][:], w["2"][:]]

        elif "embedding" == re.sub(r'[^a-zA-Z\s]', '', n):
          w = layers[n]["vars"]
          weight_layer = w["0"][:]

        elif "dropout" == re.sub(r'[^a-zA-Z\s]', '', n):
          continue

        elif "bidirectional" == re.sub(r'[^a-zA-Z\s]', '', n):
          w = layers[n]["forward_layer"]["cell"]["vars"]
          units = w["1"].shape[0]
          weight_layer = [w["0"][:], w["1"][:], w["2"][:]]

          w = layers[n]["backward_layer"]["cell"]["vars"]
          units = w["1"].shape[0]
          weight_layer.append(w["0"][:])
          weight_layer.append(w["1"][:])
          weight_layer.append(w["2"][:])

        weights[n] = weight_layer

    sorted_seq = [lay.name for lay in self.seq]
    sorted_seq = sorted(sorted_seq)

    for lay1, lay2 in zip(names, sorted_seq):
      if lay1 != lay2:
        raise ValueError(f"Arsitektur tidak sama dengan file bobot. Layer {lay1} tidak sama dengan {lay2}.")

    for key, val in weights.items():
      # masukkan weight ke masing-masing layer scratch
      for layer in self.seq:
        if key == layer.name:
          if "embedding" == re.sub(r'[^a-zA-Z\s]', '', key):
            layer.weights = [val]
            continue
          layer.set_weights(val)

    for lay in self.seq:
      if re.sub(r'[^a-zA-Z\s]', '', lay.name) != "dropout":
        self.weights = self.weights + lay.weights

    return self

  def add(self, layer):
    # apabila sudah ada model yang sama, tambahkan _count
    # add layer to seq
    count = 0
    for lay in self.seq:
      if layer.name == re.sub(r'[^a-zA-Z\s]', '', lay.name):
        count = count + 1

    if count >= 1:
      layer.name = layer.name + "_" + str(count)

    self.layer_names.append(layer.name)

    self.seq.append(layer)
    return self

  def predict(self, x, batch_size=32):
    # predict through each layer
    out = []
    batch_size = batch_size if len(x) >= batch_size else len(x)

    for batch in batch_array(x, batch_size=batch_size):
      x_batch = batch
      for lay in self.seq:
        x_batch = lay.forward(x_batch)
      out.append(x_batch)

    out = np.concatenate(out, axis=0)
    return out
    
class lstm:
  def __init__(self, units, return_seq=False, return_state=False):
    self.units = units
    self.return_seq = return_seq
    self.return_state = return_state
    self.weights = None
    self.name = "lstm"

  def set_weights(self, x):
    # ref: https://stackoverflow.com/questions/42861460/how-to-interpret-weights-in-a-lstm-layer-in-keras
    self.weights = x
    self.w = self.weights[0]  # bobot input
    self.u = self.weights[1]  # bobot recurrent
    self.b = self.weights[2]  # bobot bias

    # bobot kernel (bobot input)
    self.w_i = self.w[:, :self.units]
    self.w_f = self.w[:, self.units: self.units * 2]
    self.w_c = self.w[:, self.units * 2: self.units * 3]
    self.w_o = self.w[:, self.units * 3:]

    # bobot recurrent kernel
    self.u_i = self.u[:, :self.units]
    self.u_f = self.u[:, self.units: self.units * 2]
    self.u_c = self.u[:, self.units * 2: self.units * 3]
    self.u_o = self.u[:, self.units * 3:]

    # bobot bias
    self.b_i = self.b[:self.units]
    self.b_f = self.b[self.units: self.units * 2]
    self.b_c = self.b[self.units * 2: self.units * 3]
    self.b_o = self.b[self.units * 3:]
    return self

  def input(self, x, h):
    out = np.dot(x, self.w_i) + np.dot(h, self.u_i) + self.b_i
    i = sigmoid(out)
    return i

  def forget(self, x, h):
    out = np.dot(x, self.w_f) + np.dot(h, self.u_f) + self.b_f
    f = sigmoid(out)
    return f

  def c_tilde(self, x, h):
    out = np.dot(x, self.w_c) + np.dot(h, self.u_c) + self.b_c
    c_ti = np.tanh(out)
    return c_ti

  def output(self, x, h):
    out = np.dot(x, self.w_o) + np.dot(h, self.u_o) + self.b_o
    o = sigmoid(out)
    return o

  def c_state(self, f, c, i, c_tilde):
    c = f*c + i*c_tilde
    return c

  def h_state(self, o, c):
    h = o*np.tanh(c)
    return h

  def get_initial_state(self, batch_size):
    h = np.zeros((batch_size, self.units))
    c = np.zeros((batch_size, self.units))
    return h, c

  def forward(self, x):
    h, c = self.get_initial_state(batch_size=x.shape[0])

    c_t = np.zeros((x.shape[0], x.shape[1], self.units))
    h_t = np.zeros((x.shape[0], x.shape[1], self.units))

    seq_len = x.shape[1]

    for t in range(seq_len):
      i = self.input(x[:, t], h)
      f = self.forget(x[:, t], h)
      c_ti = self.c_tilde(x[:, t], h)
      o = self.output(x[:, t], h)

      c = self.c_state(f, c, i, c_ti)
      h = self.h_state(o, c)

      c_t[:, t, :] = c # if return_state, return c_t dan h_t
      h_t[:, t, :] = h  # if return_sequence, return h_t semua timestep

    out = h_t[:,-1,:]

    if self.return_seq == True and self.return_state == False:
      return h_t
    elif self.return_seq == True and self.return_state == True:
      return h_t, out, c_t[:,-1,:]
    elif self.return_seq == False and self.return_state == True:
      return out, out, c_t[:,-1,:]
    elif self.return_seq == False and self.return_state == False:
      return out

# class dense:
class dense:
  def __init__(self, units, activation="linear"):
    self.units = units
    self.activation = "linear" if activation is None else activation
    self.weights = None
    self.name = "dense"

  def set_weights(self, x):
    self.weights = x
    self.w = self.weights[0]
    self.b = self.weights[1]
    return self

  def forward(self, x):
    z = np.dot(x,self.w) + self.b.reshape((1,len(self.b)))

    if self.activation == "linear":
      out = z
    elif self.activation == "sigmoid":
      out = sigmoid(z)
    elif self.activation == "softmax":
      out = softmax(z)
    elif self.activation == "relu":
      out = relu(z)

    return out

# class dropout:
class dropout:
  def __init__(self, rate, training=False):
    self.rate = rate
    self.training = training
    self.r = None
    self.name = "dropout"

  def forward(self, x):  # inverted dropout
    if self.training == True:
      q = 1 - self.rate
      self.r = np.random.binomial(1, self.rate, x.shape)
      out = (1/q)*self.r*x
      return out
    else:
      return x

# class embedding:
class embedding:
  def __init__(self, input_dim, output_dim, weights=None):
    self.input_dim = input_dim  # max token
    self.output_dim = output_dim  # number of output features
    self.weights = [weights]  # (input_dim, output_dim)
    self.name = "embedding"

  def forward(self, x):
    # x: (batch, seq)
    x = np.array(x)

    if len(x.shape) == 1:
      x = np.array([x])
    elif len(x.shape) != 2:
      raise ValueError(f"x harus berupa array 1D atau 2D, didapat array {len(x.shape)}D: {x.shape}")

    if x.max() >= self.input_dim:
      raise ValueError(f"Maksimum ID yang tersedia adalah {self.input_dim-1}. ID maksimum yang ditemukan adalah {x.max()}")

    embed_out = np.array([[self.weights[0][x[i][j]] for j in range(len(x[0]))] for i in range(len(x))])
    return embed_out

class bidirectional:
  def __init__(self, layer, merge_mode="concat", backward_layer=None):
    self.layer = layer  # RNN, LSTM
    self.merge_mode = merge_mode
    self.backward_layer = copy.deepcopy(self.layer) if backward_layer is None else backward_layer # RNN/LSTM
    self.name = "bidirectional"
    self.weights = None

  def set_weights(self, x):
    self.layer.set_weights(x[0:3])
    self.backward_layer.set_weights(x[3:6])
    self.weights = self.layer.weights + self.backward_layer.weights

  def forward(self, x):
    if self.layer.return_seq == True:
      fwd = self.layer.forward(x)
      bwd = self.backward_layer.forward(x[:,::-1,:])

      if self.merge_mode == "concat":
        out = np.concatenate((fwd, bwd[:,::-1]), axis=-1)
      elif self.merge_mode == "sum":
        out = np.sum([fwd, bwd[:,::-1]], axis=0)
      elif self.merge_mode == "ave":
        out = np.mean([fwd, bwd[:,::-1]], axis=0)

      return out

    else:  # return_sequence = False
      fwd = self.layer.forward(x)
      bwd = self.backward_layer.forward(x[:,::-1,:])

      if self.merge_mode == "concat":
        out = np.concatenate((fwd, bwd), axis=1)
      elif self.merge_mode == "sum":
        out = np.sum([fwd, bwd], axis=0)
      elif self.merge_mode == "ave":
        out = np.mean([fwd, bwd], axis=0)

      return out