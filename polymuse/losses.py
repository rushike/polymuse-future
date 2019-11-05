from keras.losses import categorical_crossentropy
import keras.losses
import tensorflow as tf

def rmsecat(depth):   
    def rmsecat_(y_true, y_pred):
        a = []
        h_ = None
        for i in range(depth * 2):
            h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
            if h_ is None: h_ = tf.square(h__)
            else: h_ += tf.square(h__)
        a = (tf.sqrt(h_) / (2 * depth))
        return a
    return rmsecat_