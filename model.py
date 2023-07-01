import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import eeg_positions
import spektral

def AAD(shape_eeg, ELE, sources=2):
    """ Parameters:
        shape_eeg:     tuple, shape of EEG (channel, feature).
        ELE:           list, names of EEG electrodes in 'str'. Each element in turn corresponds to a row in EEG.
        sources:       int, number of sound sources in a mixed stimulus.
    """
    # Input
    inputs = tf.keras.Input(shape=shape_eeg)
    eeg = inputs
    
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
                distance[row,col] = np.linalg.norm(xyz[row,:] - xyz[col,:])**2
            else:
                distance[row,col] = distance[col,row]

    AdjMat0 = np.reciprocal(distance)
    AdjMat0 = (AdjMat0 - np.min(AdjMat0)) / (np.max(AdjMat0) - np.min(AdjMat0))

    AdjMat0 = spektral.utils.gcn_filter(AdjMat0, symmetric=True)
    GCN = spektral.layers.GCNConv(round(eeg.shape[2]), activation='softplus', use_bias=True, kernel_initializer='he_uniform')
    eeg = GCN([eeg, AdjMat0])

    BN0 = tf.keras.layers.BatchNormalization()
    eeg = BN0(eeg)
    
    # GAT
    NumNod = eeg.shape[1]
    AdjMat1 = tf.ones([NumNod, NumNod], tf.float32)
    AdjMat1 = tf.expand_dims(AdjMat1, 0)
    multiples = tf.stack([tf.shape(eeg)[0], tf.constant(1), tf.constant(1)], axis=0)
    AdjMat1 = tf.tile(AdjMat1, multiples)

    GAT = spektral.layers.GATConv(round(eeg.shape[2]), attn_heads=3, concat_heads=False, dropout_rate=0, return_attn_coef=True,
                                  add_self_loops=False, activation='softplus', use_bias=True, kernel_initializer='he_uniform')

    eeg, attn_coef = GAT([eeg, AdjMat1])
    attn_coef = tf.math.reduce_mean(attn_coef, axis=2, keepdims=False)    # average attention coefficient matrix of several attention heads

    BN1 = tf.keras.layers.BatchNormalization()
    eeg = BN1(eeg)
    
    # Readout
    eeg = tf.concat([spektral.layers.GlobalAvgPool()(eeg), spektral.layers.GlobalMaxPool()(eeg)], -1)
    
    # Classification
    Dense0 = tf.keras.layers.Dense(math.ceil(eeg.shape[1] /1.5), activation='softplus', kernel_initializer='he_uniform')
    eeg = Dense0(eeg)

    Dense1 = tf.keras.layers.Dense(math.ceil(eeg.shape[1] /2), activation='softplus', kernel_initializer='he_uniform')
    eeg = Dense1(eeg)

    Softmax = tf.keras.layers.Dense(sources, activation='softmax', name='label')
    label = Softmax(eeg)
    
    # Building a model
    model = tf.keras.Model(inputs=inputs, outputs=[label, attn_coef], name='AAD-GCN-GAT')
    model.compile(loss=['sparse_categorical_crossentropy', 'mean_squared_error'],
                  loss_weights=[1, 0],    # attn_coef is only output; it does not involve backpropagation
                  optimizer=tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-7, epsilon=1e-3, amsgrad=True),
                  metrics=[['sparse_categorical_accuracy'], ['mean_squared_error']])
    model.summary()
    return model
