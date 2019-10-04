
import numpy

"""

 1

-----

 1   0
-    1


-----

 1  -1    0
-    1   -1

-----

 1  -2    1    0
-    1   -2    1

------

 1   -3    3    -1    0
-     1   -3     3   -1

------

 1   -4    6    -4    1    0
-     1   -4     6   -4    1  
     
------


 1   -5     10    -10    5     -1




General f^{n}_{r} = (-1) ^ r  ^{n}C_{r}    ...  0 <= r <= n

where f^{n} is derivative polynomial

"""

"""
Derivate of only 3D array and only along axis 1(middle axis)
"""

# def first_derivative(roll_mat, axis = 1):
#     """
#     find the derivative of number numpy ndarray
#     """

#     res_set = numpy.zeros(roll_mat.shape)
#     last = numpy.array([0, 0, 0 ,0, 0])
#     for j in range(roll_mat.shape[0]):
#         for i in range(1, roll_mat.shape[1]):
#             if roll_mat[j, i, 0] == 0: continue
#             last = roll_mat[j, i]
#             for k in range(roll_mat.shape[2]):
#                 if roll_mat[j, i, k] == 0 or roll_mat[j, i - 1, k] == 0: break
#                 res_set[j, i, k]  = last[k] - roll_mat[j, i - 1, k]
            
#     return res_set


def first_derivative(ns_array_repr):
    """
    magenta ns array repr to its derivative

    [n2 - n1, velo / duration] 

    """

    #ns validity check

    shape = ns_array_repr.shape

    if shape[1] != 4 : raise AttributeError("ns_array_repr not in format")

    derv = numpy.zeros(shape[:1] + (3,))

    for i in range(1, shape[0]):
        derv[i] = [ns_array_repr[i][0] ,ns_array_repr[i][1] - ns_array_repr[i - 1][1], ns_array_repr[i][2] / ns_array_repr[i][3]] 

    return derv