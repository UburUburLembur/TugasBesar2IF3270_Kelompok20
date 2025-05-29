# src/rnn/model.py

import h5py
import numpy as np
import os
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def _load_weights_recursive(h5_item):
    weights = {}
    if isinstance(h5_item, h5py.Group):
        for name_key in h5_item.keys():
            item = h5_item[name_key]
            if isinstance(item, h5py.Dataset):
                weights[name_key] = item[()]
            elif isinstance(item, h5py.Group):
                weights[name_key] = _load_weights_recursive(item)
    elif isinstance(h5_item, h5py.Dataset):
        return h5_item[()]
    return weights

def load_weights_from_hdf5(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File bobot tidak ditemukan di: {filepath}")
    with h5py.File(filepath, 'r') as hf:
        return _load_weights_recursive(hf)

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim, name=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.name = name
        self.embedding_matrix = None 
        self._is_built = False

    def set_weights(self, weights_data):
        actual_matrix = None
        if isinstance(weights_data, np.ndarray):
            actual_matrix = weights_data
        elif isinstance(weights_data, dict):
            potential_key_direct = next((k for k in weights_data if 'embeddings' in k or k == 'weight'), None)
            if potential_key_direct and isinstance(weights_data.get(potential_key_direct), np.ndarray):
                actual_matrix = weights_data[potential_key_direct]
            elif 'vars' in weights_data and isinstance(weights_data.get('vars'), dict):
                vars_dict = weights_data['vars']
                if '0' in vars_dict and isinstance(vars_dict.get('0'), np.ndarray):
                    actual_matrix = vars_dict['0']
                else: 
                    potential_key_in_vars = next((k for k in vars_dict if 'embeddings' in k or k == 'weight'), None)
                    if potential_key_in_vars and isinstance(vars_dict.get(potential_key_in_vars), np.ndarray):
                        actual_matrix = vars_dict[potential_key_in_vars]
            elif len(weights_data) == 1:
                first_value = next(iter(weights_data.values()))
                if isinstance(first_value, np.ndarray):
                    actual_matrix = first_value
        
        if actual_matrix is None:
            raise ValueError(f"Tidak dapat mengekstrak matriks embedding untuk layer '{self.name}' dari data: {list(weights_data.keys() if isinstance(weights_data,dict) else 'N/A')}")

        # Validasi shape
        if actual_matrix.shape[0] != self.vocab_size or actual_matrix.shape[1] != self.embedding_dim:
            raise ValueError(f"Shape matriks embedding ({actual_matrix.shape}) untuk layer '{self.name}' tidak cocok "
                             f"dengan konfigurasi ({self.vocab_size}, {self.embedding_dim}).")
        self.embedding_matrix = actual_matrix
        self._is_built = True

    def forward(self, inputs):
        """Melakukan forward pass."""
        if not self._is_built or self.embedding_matrix is None:
            raise RuntimeError(f"Bobot untuk EmbeddingLayer '{self.name}' belum dimuat. Panggil load_weights pada model.")
        
        if not isinstance(inputs, np.ndarray): inputs = np.array(inputs)
        if inputs.ndim == 0: inputs = np.expand_dims(inputs, axis=0)

        if np.any(inputs >= self.embedding_matrix.shape[0]) or np.any(inputs < 0):
            raise ValueError(f"Indeks input ({np.min(inputs)}-{np.max(inputs)}) di luar jangkauan embedding matrix untuk '{self.name}' (0-{self.embedding_matrix.shape[0]-1}). Shape matriks: {self.embedding_matrix.shape}")
        
        if inputs.ndim == 1: return self.embedding_matrix[inputs]
        elif inputs.ndim == 2:
            batch_size, seq_len = inputs.shape
            output = np.zeros((batch_size, seq_len, self.embedding_dim))
            for i in range(batch_size): output[i] = self.embedding_matrix[inputs[i]]
            return output
        else: raise ValueError("Input Embedding harus 1D atau 2D.")


class SimpleRNNLayer:
    def __init__(self, units, activation='tanh', return_sequences=False, name=None):
        self.units = units
        self.name = name
        if isinstance(activation, str):
            if activation == 'tanh': self.activation_fn = tanh
            elif activation == 'sigmoid': self.activation_fn = sigmoid
            elif activation == 'relu': self.activation_fn = relu
            else: raise ValueError(f"Aktivasi string tidak dikenal: {activation}")
        else: self.activation_fn = activation
        self.return_sequences = return_sequences
        self.kernel, self.recurrent_kernel, self.bias = None, None, None
        self._is_built = False

    def set_weights(self, weights_data):
        k, rk, b = None, None, None
        k = weights_data.get('kernel') or weights_data.get('kernel:0')
        rk = weights_data.get('recurrent_kernel') or weights_data.get('recurrent_kernel:0')
        b = weights_data.get('bias') or weights_data.get('bias:0')

        if k is None and 'cell' in weights_data and isinstance(weights_data.get('cell'), dict) and \
           'vars' in weights_data['cell'] and isinstance(weights_data['cell'].get('vars'), dict):
            cell_vars = weights_data['cell']['vars']
            k, rk, b = cell_vars.get('0'), cell_vars.get('1'), cell_vars.get('2')
        
        if k is None or rk is None or b is None:
            raise ValueError(f"Bobot kernel/recurrent_kernel/bias untuk layer '{self.name}' tidak ditemukan atau tidak lengkap di data: {list(weights_data.keys()) if isinstance(weights_data,dict) else 'N/A'}")

        if not (k.shape[1] == self.units and rk.shape[0] == self.units and \
                rk.shape[1] == self.units and b.shape[0] == self.units):
            raise ValueError(f"Shape bobot untuk '{self.name}' tidak cocok dengan units={self.units}. K:{k.shape}, RK:{rk.shape}, B:{b.shape}")
        
        self.kernel, self.recurrent_kernel, self.bias = k, rk, b
        self._is_built = True

    def forward(self, inputs, initial_state=None):
        """Melakukan forward pass."""
        if not self._is_built: raise RuntimeError(f"Bobot untuk SimpleRNNLayer '{self.name}' belum dimuat.")
        
        if inputs.ndim != 3: raise ValueError(f"Input SimpleRNN '{self.name}' harus 3D (batch_size, timesteps, input_dim). Diterima: {inputs.shape}")
        batch_size, timesteps, input_dim = inputs.shape

        if self.kernel.shape[0] != input_dim: raise ValueError(f"Dimensi input kernel ({self.kernel.shape[0]}) di '{self.name}' tidak cocok dengan input_dim ({input_dim}).")
        
        h_t = np.zeros((batch_size, self.units)) if initial_state is None else initial_state
        outputs_sequence = []
        for t in range(timesteps):
            x_t = inputs[:, t, :]
            h_t = self.activation_fn(np.dot(x_t, self.kernel) + np.dot(h_t, self.recurrent_kernel) + self.bias)
            if self.return_sequences: outputs_sequence.append(h_t)
        
        return np.stack(outputs_sequence, axis=1) if self.return_sequences else h_t



class BidirectionalSimpleRNNLayer:
    def __init__(self, units, activation='tanh', return_sequences=False, merge_mode='concat', name=None):
        self.units = units
        self.name = name
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode
        self.forward_rnn = SimpleRNNLayer(units, activation, return_sequences, name=f"{name}_forward_rnn_internal")
        self.backward_rnn = SimpleRNNLayer(units, activation, return_sequences, name=f"{name}_backward_rnn_internal")
        self._is_built = False

    def set_weights(self, weights_data):
        fwd_key, bwd_key = None, None
        for k_dict in weights_data: 
            if 'forward' in k_dict.lower(): fwd_key = k_dict
            elif 'backward' in k_dict.lower(): bwd_key = k_dict
        
        if not fwd_key or not bwd_key:
            raise ValueError(f"Kunci forward/backward layer tidak ditemukan di '{self.name}' dari data: {list(weights_data.keys())}")

        self.forward_rnn.set_weights(weights_data[fwd_key])
        self.backward_rnn.set_weights(weights_data[bwd_key])
        self._is_built = True

    def forward(self, inputs):
        """Melakukan forward pass."""
        if not self._is_built: raise RuntimeError(f"Bobot untuk BidirectionalSimpleRNNLayer '{self.name}' belum dimuat.")
        
        output_forward = self.forward_rnn.forward(inputs)
        inputs_reversed = np.flip(inputs, axis=1) 
        output_backward_reversed = self.backward_rnn.forward(inputs_reversed)
        
        output_backward = np.flip(output_backward_reversed, axis=1) if self.return_sequences else output_backward_reversed
        
        if self.merge_mode == 'concat': return np.concatenate((output_forward, output_backward), axis=-1)
        elif self.merge_mode == 'sum': return output_forward + output_backward
        else: raise ValueError(f"Mode penggabungan '{self.merge_mode}' tidak didukung di '{self.name}'.")


class DenseLayer:
    def __init__(self, units, activation=None, name=None):
        self.units = units
        self.name = name
        if isinstance(activation, str):
            if activation == 'tanh': self.activation_fn = tanh
            elif activation == 'sigmoid': self.activation_fn = sigmoid
            elif activation == 'relu': self.activation_fn = relu
            elif activation == 'softmax': self.activation_fn = softmax
            elif activation is None or activation == 'linear': self.activation_fn = None
            else: raise ValueError(f"Aktivasi string tidak dikenal: {activation}")
        elif callable(activation): self.activation_fn = activation
        elif activation is not None: raise ValueError(f"Tipe aktivasi tidak valid: {type(activation)}")
        else: self.activation_fn = None
        self.kernel, self.bias = None, None
        self._is_built = False

    def set_weights(self, weights_data):
        k, b = None, None
        k = weights_data.get('kernel') or weights_data.get('kernel:0')
        b = weights_data.get('bias') or weights_data.get('bias:0')
        if k is None and 'vars' in weights_data and isinstance(weights_data.get('vars'), dict):
            layer_vars = weights_data['vars']
            k, b = layer_vars.get('0'), layer_vars.get('1')

        if k is None or b is None:
            raise ValueError(f"Bobot kernel/bias untuk layer '{self.name}' tidak ditemukan atau tidak lengkap di data: {list(weights_data.keys()) if isinstance(weights_data,dict) else 'N/A'}")
        
        if k.shape[1] != self.units or b.shape[0] != self.units:
             raise ValueError(f"Shape bobot untuk '{self.name}' tidak cocok units={self.units}. K:{k.shape}, B:{b.shape}")

        self.kernel, self.bias = k, b
        self._is_built = True
        
    def forward(self, inputs):
        """Melakukan forward pass."""
        if not self._is_built: raise RuntimeError(f"Bobot untuk DenseLayer '{self.name}' belum dimuat.")
        
        if inputs.shape[-1] != self.kernel.shape[0]: raise ValueError(f"Dimensi input Dense ({inputs.shape[-1]}) di '{self.name}' tidak cocok dengan dimensi input kernel ({self.kernel.shape[0]}).")
        
        if inputs.ndim == 3: 
            output = np.einsum('btf,fu->btu', inputs, self.kernel) + self.bias
        elif inputs.ndim == 2:
            output = np.dot(inputs, self.kernel) + self.bias
        else: raise ValueError(f"Input Dense '{self.name}' harus 2D atau 3D. Diterima shape: {inputs.shape}")
        
        return self.activation_fn(output) if self.activation_fn else output


class DropoutLayer: 
    def __init__(self, rate, name=None): 
        self.rate = rate
        self.name = name if name else f"dropout_{np.random.randint(1000)}"

    def set_weights(self, weights_data): 
        pass 

    def forward(self, inputs, training=False):
        if training: 
            mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            return inputs * mask
        return inputs


class Model:
    def __init__(self, name=None):
        self.layers = []
        self.name = name if name else "MySequentialModel"
        self.auto_name_counts = {} 

    def add(self, layer):
        user_provided_name = getattr(layer, 'name', None)
        final_name = user_provided_name

        if final_name is None:
            layer_class_name = layer.__class__.__name__
            base_name = ""

            if isinstance(layer, EmbeddingLayer):
                base_name = "embedding"
            elif isinstance(layer, SimpleRNNLayer) and not isinstance(layer, BidirectionalSimpleRNNLayer):
                base_name = "simple_rnn"
            elif isinstance(layer, BidirectionalSimpleRNNLayer):
                base_name = "bidirectional"
            elif isinstance(layer, DenseLayer):
                base_name = "dense"
            elif isinstance(layer, DropoutLayer):
                base_name = "dropout"
            else:
                base_name = layer_class_name.lower().replace("layer", "")
            
            current_count = self.auto_name_counts.get(base_name, 0)

            if current_count == 0:
                final_name = base_name
            else:
                final_name = f"{base_name}_{current_count}"
            
            layer.name = final_name
            self.auto_name_counts[base_name] = current_count + 1

        self.layers.append(layer)

    def load_weights(self, filepath):
        all_h5_weights = load_weights_from_hdf5(filepath)

        for layer in self.layers:
            if not hasattr(layer, 'set_weights'):
                print(f"  Layer '{getattr(layer, 'name', layer.__class__.__name__)}' ({layer.__class__.__name__}) tidak memiliki metode 'set_weights', dilewati.")
                continue
            
            if isinstance(layer, DropoutLayer):
                if hasattr(layer, 'set_weights'): layer.set_weights(None)
                continue

            if not layer.name:
                raise ValueError(f"Layer {layer} tidak memiliki atribut 'name' setelah ditambahkan, tidak bisa memuat bobot.")

            weights_data_for_this_layer = all_h5_weights["layers"].get(layer.name)

            if weights_data_for_this_layer is None:
                raise ValueError(
                    f"Bobot untuk layer '{layer.name}' tidak ditemukan sebagai kunci top-level "
                    f"di file HDF5 ('{filepath}'). "
                    f"Kunci top-level yang tersedia: {list(all_h5_weights.keys())}"
                )
            
            try:
                layer.set_weights(weights_data_for_this_layer)
            except Exception as e:
                print(f"ERROR saat memuat bobot untuk layer '{layer.name}': {e}")
                raise 

    def forward(self, inputs):
        x = inputs
        total_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            x = layer.forward(x)

            progress_percentage = (i + 1) * 100 / total_layers
            bar_length = 40  
            filled_length = int(bar_length * (i + 1) // total_layers)
            bar_display = '█' * filled_length + '-' * (bar_length - filled_length)
            
            sys.stdout.write(f'\rProcessing Layers: |{bar_display}| {progress_percentage:.2f}%')
            sys.stdout.flush() 

        sys.stdout.write('\n') 
        return x

    def summary(self):
        print(f"\n--- Ringkasan Model: '{self.name}' ---")
        total_params = 0
        print("_________________________________________________________________")
        print("Layer (type)                 Output Shape              Param #   ")
        print("=================================================================")
        for layer in self.layers:
            layer_name_str = layer.name if hasattr(layer, 'name') and layer.name else layer.__class__.__name__
            layer_type_str = layer.__class__.__name__
            output_shape_str = "(Variable)"
            
            params_count = 0
            if hasattr(layer, 'kernel') and layer.kernel is not None: params_count += np.prod(layer.kernel.shape)
            if hasattr(layer, 'bias') and layer.bias is not None: params_count += np.prod(layer.bias.shape)
            if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None: params_count += np.prod(layer.recurrent_kernel.shape)
            if hasattr(layer, 'embedding_matrix') and layer.embedding_matrix is not None: params_count += np.prod(layer.embedding_matrix.shape)
            
            if isinstance(layer, BidirectionalSimpleRNNLayer):
                params_count = 0 
                if hasattr(layer.forward_rnn, 'kernel') and layer.forward_rnn.kernel is not None: params_count += np.prod(layer.forward_rnn.kernel.shape)
                if hasattr(layer.forward_rnn, 'bias') and layer.forward_rnn.bias is not None: params_count += np.prod(layer.forward_rnn.bias.shape)
                if hasattr(layer.forward_rnn, 'recurrent_kernel') and layer.forward_rnn.recurrent_kernel is not None: params_count += np.prod(layer.forward_rnn.recurrent_kernel.shape)
                if hasattr(layer.backward_rnn, 'kernel') and layer.backward_rnn.kernel is not None: params_count += np.prod(layer.backward_rnn.kernel.shape)
                if hasattr(layer.backward_rnn, 'bias') and layer.backward_rnn.bias is not None: params_count += np.prod(layer.backward_rnn.bias.shape)
                if hasattr(layer.backward_rnn, 'recurrent_kernel') and layer.backward_rnn.recurrent_kernel is not None: params_count += np.prod(layer.backward_rnn.recurrent_kernel.shape)

            total_params += params_count
            params_str = str(params_count) if params_count > 0 else "0"

            print(f"{layer_name_str[:28]:<29} {output_shape_str:<26} {params_str:<10}")
        print("=================================================================")
        print(f"Total params: {total_params}")
        print("_________________________________________________________________")