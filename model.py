import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import eeg_positions
import spektral

def squareplus(x):
    """
    Activation function
    """
    b = 4
    return 0.5 * (tf.math.sqrt(tf.math.pow(x, 2) + b) + x)

class Aij(tf.keras.layers.Layer):
    """
    Attention coefficient matrix in GAT, referring to
    Petar Veličković et al. Graph Attention Networks 2018 ICLR
    """
    def __init__(self):
        super(Aij, self).__init__(trainable=True)

    def build(self, input_shape):
        features = input_shape[2]

        attn_heads = 1
        self.kernel = self.add_weight(name="kernel", shape=[features, attn_heads, features])
        self.attn_kernel_self = self.add_weight(name="attn_kernel_self", shape=[features, attn_heads, 1])
        self.attn_kernel_neighs = self.add_weight(name="attn_kernel_neighs", shape=[features, attn_heads, 1])

        super(Aij, self).build(input_shape)

    def call(self, encode):
        encode = tf.einsum("...NI , IHO -> ...NHO", encode, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", encode, self.attn_kernel_self)
        attn_for_neighs = tf.einsum("...NHI , IHO -> ...NHO", encode, self.attn_kernel_neighs)
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)

        attn_coef = tf.math.reduce_mean(attn_coef, axis=2, keepdims=False)
        return attn_coef
    
def ASAD(shape_eeg, ELE, sources=2):
    """ Parameters:
        shape_eeg:     tuple, shape of EEG (channel, feature).
        ELE:           list, names of EEG electrodes in 'str'. Each element in turn corresponds to a row in EEG.
        sources:       int, number of sound sources in a mixed stimulus.
    """
    # Input
    inputs = tf.keras.Input(shape=shape_eeg)
    eeg = inputs

    boundary = int(shape_eeg[1] /1.5)
    DE, HOC = tf.split(eeg, [boundary,tf.shape(eeg)[2]-boundary], axis=-1)

    LN_DE = tf.keras.layers.LayerNormalization(axis=-2)    # Layer normalization across EEG channel axis
    DE = LN_DE(DE)

    LN_HOC = tf.keras.layers.LayerNormalization(axis=-2)
    HOC = LN_HOC(HOC)

    eeg = tf.keras.layers.concatenate([DE, HOC], axis=-1)
    
    # GCN
    coordinate = eeg_positions.get_elec_coords(elec_names=ELE, dim='3d')
    x = np.expand_dims(coordinate['x'].values, axis=-1).astype('float32')
    y = np.expand_dims(coordinate['y'].values, axis=-1).astype('float32')
    z = np.expand_dims(coordinate['z'].values, axis=-1).astype('float32')
    xyz = np.concatenate((x,y,z), axis=-1)    # 3D coordinates of the EEG electrodes

    distance = np.zeros((len(ELE),len(ELE)), dtype='float32')    # Euclidean distance between every two EEG electrodes
    for row in range(len(ELE)):
        for col in range(len(ELE)):
            if row == col:
                distance[row,col] = np.inf
            elif col > row:
                distance[row,col] = np.linalg.norm(xyz[row,:] - xyz[col,:])    # dij
            else:
                distance[row,col] = distance[col,row]

    AdjMat = np.reciprocal(np.square(distance))    # Aij = 1 / dij**2    Aii = 0
    AdjMat = (AdjMat - np.min(AdjMat)) / (np.max(AdjMat) - np.min(AdjMat))    # Min-max normalization

    AdjMat = spektral.utils.gcn_filter(AdjMat, symmetric=True)
    GCN = spektral.layers.GCNConv(eeg.shape[2], activation=squareplus, use_bias=True, kernel_initializer='he_normal')
    eeg = GCN([eeg, AdjMat])

    BN = tf.keras.layers.BatchNormalization()    # Batch-wise graph normalization
    eeg = BN(eeg)
    
    # Global attention-based readout
    attn_coef = Aij()(eeg)    # global attention coefficients
    Attn_coef = tf.math.reduce_mean(attn_coef, axis=1)    # Weights

    Attn_coef = tf.expand_dims(Attn_coef, -1)
    Attn_coef = tf.tile(Attn_coef, tf.constant([1,1,eeg.shape[2]]))
    eeg = tf.math.multiply_no_nan(Attn_coef, eeg)
    eeg = tf.math.reduce_sum(eeg, axis=1)
    
    # Classification
    Dense = tf.keras.layers.Dense(math.ceil(eeg.shape[1] /2), activation=squareplus, kernel_initializer='he_normal')
    eeg = Dense(eeg)

    Softmax = tf.keras.layers.Dense(sources, activation='softmax', name='label')
    label = Softmax(eeg)
    
    # Building a model
    model = tf.keras.Model(inputs=inputs, outputs=[label, attn_coef])
    model.compile(loss=['sparse_categorical_crossentropy', []],
                  loss_weights=[1, 0],    # attn_coef does not involve backpropagation
                  optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-3, amsgrad=True),
                  metrics=[['sparse_categorical_accuracy'], []])
    model.summary()
    return model
