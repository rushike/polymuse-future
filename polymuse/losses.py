from keras.losses import categorical_crossentropy
import keras.losses
import tensorflow as tf

# def rmsecat(depth):   
#     def rmsecat_(y_true, y_pred):
#         a = []
#         h_ = None
#         for i in range(depth * 2):
#             h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
#             if h_ is None: h_ = tf.square(h__)
#             else: h_ += tf.square(h__)
#         a = (tf.sqrt(h_) / (2 * depth))
#         return a
#     return rmsecat_


def rmsecat2(depth):   
    def rmsecat_(y_true, y_pred):
        a = []
        h_ = None
        y_true = y_true + 2e-2
        lg = y_true * tf.math.log(y_pred)
        for i in range(depth * 2):
            sm = tf.math.reduce_sum(lg[:, i : i + 16], axis= 1)
            # h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
            
            if h_ is None: h_ = tf.square(sm)
            else: h_ += tf.square(sm)
        a = (tf.sqrt(h_) / (2 * depth))
        return a
    return rmsecat_


def rmsecat(depth):   
    def rmsecat_(y_true, y_pred):
        a = []
        h_ = None
        dif = tf.abs(y_true - y_pred)
        lg = dif * tf.math.log(1 - dif)
        for i in range(depth * 2):
            sm = tf.math.reduce_sum(lg[:, i : i + 16], axis= 1)
            # h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
            
            if h_ is None: h_ = tf.square(sm)
            else: h_ += tf.square(sm)
        a = (tf.sqrt(h_) / (2 * depth))
        return a
    return rmsecat_




# def rmscate(depth):   
#     def rmscate_(y_true, y_pred):
#         a = []
#         h_ = None
#         y_true = y_true + 1e-2
#         lg = y_true * tf.math.log(y_pred)
#         for i in range(depth * 2):
#             sm = tf.math.reduce_sum(lg[:, i : i + 16], axis= 1)
#             # h__ = categorical_crossentropy(y_true[:, i : i + 16], y_pred[ :, i : i + 16]) 
            
#             if h_ is None: h_ = tf.square(sm)
#             else: h_ += tf.square(sm)
#         a = (tf.sqrt(h_) / (2 * depth))
#         return a
#     return rmscate_

