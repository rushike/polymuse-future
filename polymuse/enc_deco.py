import numpy

def sFlat_to_octave(sflat, depth = 16):
    """sFlat to octave representation
    converts each element to 2 * 16 matrix
    0th row for octave, 1st row for note

    off note encoded as 15 

    C  C#  D D#  E  F  F# G  G# A  A#  B  
    0  1   2  3  4  5  6  7  8  9  10  11
    Arguments:
        sflat {sFlat-> numpy.ndarray} -- [description]
    
    Keyword Arguments:
        depth {int} -- [description] (default: {16})
    """
    octave = numpy.zeros(sflat.shape + (2, depth))
    print("oct sh : ", octave.shape)
    for i in range(sflat.shape[0]):
        for j in range(sflat.shape[1]):
            for k in range(sflat.shape[2]):
                octv, note = sflat[i, j, k] // 12, sflat[i, j, k] % 12
                if sflat[i, j, k] == 0: octv, note = 0, 15
                # print("cat octv : ", to_categorical(octv, depth))
                octave[i, j, k, :1] = to_categorical(octv, depth)
                octave[i, j, k, 1:] = to_categorical(note, depth)

    return octave

def octave_to_sFlat(octave):
    sflat = numpy.zeros(octave.shape[:-2])
    for i in range(sflat.shape[0]):
        for j in range(sflat.shape[1]):
            for k in range(sflat.shape[2]):
                octv, note = numpy.argmax(octave[i, j, k, 0]), numpy.argmax(octave[i, j, k, 1]) 
                
                sflat[i, j, k] = octv * 12 + note 
                if octv == 0 and note == 15:
                    sflat[i, j, k] = 0
    return sflat

def tm_to_enc_tm(tm, classes = 64, MAX = 4):
    enc_tm = numpy.zeros(tm.shape[:-1] + (classes, ))
    # print(tm)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            for k in range(tm.shape[2]):
                y = int(tm[i, j, k] * MAX / 4)
                # print(y)
                enc_tm[i, j] = to_categorical(y, classes)
    return enc_tm

def enc_tm_to_tm(tm, MAX = 4):
    res_tm = numpy.zeros(tm.shape[:-1] + (1, ))

    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            # for k in range(tm.shape[2]):
            y = numpy.argmax(tm[i, j]) 
            res_tm[i, j, 0] = y / MAX  * 4
    return res_tm

def thres_octave(tm):
    pass

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = numpy.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = numpy.max(y) + 1
    n = y.shape[0]
    categorical = numpy.zeros((n, num_classes), dtype=dtype)
    categorical[numpy.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = numpy.reshape(categorical, output_shape)
    
    return categorical
