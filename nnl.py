#################################################################
# Code      NN via MM Library
# Version   1.0
# Date      2025-09-05
# Author    Dan Humfeld, DanHumfeld@GraceECLLC.com
# Note      This library enables the training and use of NNs
#           including PINNs.
#
#################################################################
# Importing libraries 
#################################################################
import random
import math

import sys      # Used for debugging only at this point

#################################################################
# Fundamental Classes
#################################################################
class Model:
    ''' Class to hold an entire model and perform calculcations '''

    def __init__(self, inputs:int):
        self.nodes = [inputs]
        self.activation_functions = ['']
        self.optimizer = ''

        self.w = [[]]           # The weights is the only list that is not one-per-residual, and is the only object saved or loaded
        self.x = []
        self.y = []
        self.z = []
        self.q = []
        self.v = []
        self.p = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        self.f = []
        self.calculate_1st_list = []
        self.calculate_2nd_list = []
    
    def add_layer(self, nodes:int, activation_function:str):
        self.nodes.append(nodes)
        self.activation_functions.append(activation_function)
    
    def compile(self, test_mode = False):
        self.w = [[[0]]]
        for layer in range(len(self.nodes)-1):            # For n layers, this is 0..(n-1), so it defines the weights that pass from this layer to the next layer, which is not what is used in the derivation. Correcting by adding self.w=[[0]].
            w_layer = [[random.random() for input_node in range(self.nodes[layer])] for output_node in range(self.nodes[layer+1])]            # Order is correct for "layer-to-from".
            self.w.append(w_layer)
        # Example: if self.nodes = [3,5,5,1] then len(self.nodes) = 4, w has a 0th entry and then gets values for 1 2 3, which is rght. Layer 0 are the inputs. w[1] are the weights from layer 0-to-1, w[2] are from 1-to-2 and w[3] are from 2-to-3.
        if test_mode:
            #self.w = [[[0]]]
            self.calculate_1st_list = [0, 1, 2]
            self.calculate_2nd_list = [1]

    def set_optimizer(self, optimizer:str, *args):
        self.optimizer = optimizer

    def _clean_up(self):
        self.x = []
        self.y = []
        self.z = []
        self.q = []
        self.v = []
        self.p = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        self.f = []

    def apply_gradient(self):
        pass
        self._clean_up()

    def load(self, filename:str):
        pass

    def save(self, filename:str):
        pass

    def dx(self):
        pass

    def d2x(self):
        pass

    def dw(self):
        pass

    def evaluate(self, residuals):
        pass

    def calculate_xy(self, residuals):
        pass

    def calculate_data(self, residuals):
        pass

    def make_next_data_layers(self):
        self.x.append([])
        self.y.append([])
        self.z.append([])
        self.q.append([])
        self.p.append([])
        self.v.append([])
        self.a.append([])
        self.b.append([])
        self.c.append([])
        self.d.append([])
        self.e.append([])
        self.f.append([])

    def calculate_required_data(self, residuals):
        for j in range(len(self.nodes)):
            print('Processing layer', j)
            self.make_next_data_layers()
            if (j == 0):
                self.x[j] = []
                #self.y[j] = residuals
                self.y[j] = [list(row) for row in zip(*residuals)]       # This means "residuals" is a list of batch entries, each of which has length nodes[0] i.e. the model input count.   This line transposes the residuals from [res][i] to [i][res]
                self.z[j] = []      # z is only ever used in-level, for terms that are 0 for level 0
                #self.z[j] = [[1-self.y[j][i][res]**2 for res in range(len(self.y[j][i]))] for i in range(len(self.y[j]))]
                self.q[j] = []      # q is only ever used in-level, never looking back one level. q is only used in-level for terms that are 0 for level 0.
                self.p[j] = [[[
                    1 * (k==i) 
                    for res in range(len(self.y[j][i]))] 
                    for i in range(len(self.y[j]))] 
                    for k in range(len(self.y[j]))]
                self.v[j] = [[[[[      # There are no weights going into level 0, so the derivative of the model inputs with repsect to all weights is zero.
                    0 
                    for res in range(len(self.y[j][i]))]
                    for l in range(len(self.w[n][m]))] 
                    for m in range(len(self.w[n]))] 
                    for n in range(len(self.w))] 
                    for i in range(len(self.y[j]))]

                self.a[j] = [[[
                    0
                    for res in range(len(self.y[j][i]))]
                    for i in range(len(self.y[j]))]
                    if k in self.calculate_2nd_list else 0
                    for k in range(len(self.y[0]))]

                self.d[j] = []      # def are only used in-layer for d/dw (d/dy and d2/dy2), which are all zero so def don't need to be calculated.
                self.e[j] = []
                self.f[j] = []
                
                self.b[j] = [[[[[[
                    0
                    for res in range(len(self.y[j][i]))]
                    for l in range(len(self.w[n][m]))]
                    for m in range(len(self.w[n]))]
                    for n in range(len(self.w))]
                    for i in range(len(self.y[j]))]
                    if ((k in self.calculate_1st_list) or (k in self.calculate_2nd_list)) else 0
                    for k in range(len(self.y[0]))]
                
                self.c[j] = [[[[[[
                    0
                    for res in range(len(self.y[j][i]))]
                    for l in range(len(self.w[n][m]))]
                    for m in range(len(self.w[n]))]
                    for n in range(len(self.w))]
                    for i in range(len(self.y[j]))]
                    if k in self.calculate_2nd_list else 0
                    for k in range(len(self.y[0]))]

            else:
                self.x[j] = [[
                    sum([self.w[j][i][r] * self.y[j-1][r][res] for r in range(len(self.y[j-1]))])
                    for res in range(len(self.y[j-1][0]))]                  # self.y[j] isn't defined yet! Use self.y[j-1], and then use [0] becuase self.y[j-1] may not have the same length as w[j], but all y[j][i] will have the same length = residual count
                    for i in range(len(self.w[j]))]
                self.y[j] = [[
                    math.tanh(self.x[j][i][res])
                    for res in range(len(self.x[j][i]))]
                    for i in range(len(self.x[j]))]
                self.z[j] = [[
                    1-self.y[j][i][res]**2 
                    for res in range(len(self.y[j][i]))]
                    for i in range(len(self.y[j]))]
                self.q[j] = [[[
                    self.z[j][i][res] * self.w[j][i][l] 
                    for res in range(len(self.z[j][i]))]
                    for i in range(len(self.y[j]))] 
                    for l in range(len(self.y[j-1]))]
                self.p[j] = [[[
                    sum([self.q[j][l][i][res] * self.p[j-1][k][l][res] for l in range(len(self.w[j][i]))])
                    for res in range(len(self.y[j][i]))]
                    for i in range(len(self.y[j]))] 
                    if ((k in self.calculate_1st_list) or (k in self.calculate_2nd_list)) else 0
                    for k in range(len(self.y[0]))]
                #block1 = [[0 for l in range(len(self.w[n][m]))] for m in range(len(self.w[n]))]
                #block2 = [[self.z[j][i] * self.y[j-1][l] * (1 if (i==m) else 0) for l in range(len(self.y[j-1]))] for m in range(len(self.w[n]))]
                #block3 = [[sum([self.q[j][i][r] * self.v[j-1][r][n][m][l] for r in range(len(self.q[j][i]))]) for l in range(len(self.w[n][m]))] for m in range(len(self.w[n]))]
                #self.v[j] = [[1 if (n > j) else 2 if (n == j) else 3 for n in range(len(self.w))] for i in range(len(self.y[j]))]
                # Checking consistency with w. 
                # If layers is [3,5,5,1] then j_max = 3. n_max = 3. v_jinml = v_3i3ml = dy3i/dw3ml, where n=3 is the 2-to-3 layer weights. This is correct.
                # What calcualtion happens for n = 0 because w[0] = [[0]]? len(self.w[n]) = 1, len(self.w[n][m]) = 1. self.v[j-1][r][0][m][l] would be for m=0 l=0 so self.v[j-1][r][0][0][0] which all exists and they are 0-valued.
                self.v[j] = [[
                    [0] if (n == 0) else

                    [[[
                        0 
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))] 
                        for m in range(len(self.w[n]))] if (n > j) else 

                    [[[
                        self.z[j][i][res] * self.y[j-1][l][res] * (i==m) 
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.y[j-1]))] if (m==i) else
                     [[
                        0
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.y[j-1]))]       # m != i so there's no z*y term.
                        for m in range(len(self.w[n]))] if (n == j) else 

                    [[[
                        sum([self.q[j][i][r][res] * self.v[j-1][r][n][m][l][res] for r in range(len(self.q[j][i]))]) 
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))] 
                        for m in range(len(self.w[n]))] 

                    for n in range(len(self.w))] 
                    for i in range(len(self.y[j]))]
                
                self.a[j] = [[[
                    -2 * self.y[j][i][res] * self.z[j][i][res] * (sum([self.w[j][i][m] * self.p[j-1][k][m][res] for m in range(len(self.w[j][i]))]))**2 \
                    + self.z[j][i][res] * (sum([self.w[j][i][m] * self.a[j-1][k][m][res] for m in range(len(self.w[j][i]))]))
                    for res in range(len(self.y[j][i]))]
                    for i in range(len(self.y[j]))]
                    if k in self.calculate_2nd_list else 0
                    for k in range(len(self.y[0]))]

                if ((len(self.calculate_1st_list) > 0) or (len(self.calculate_2nd_list) > 0)):
                    self.d[j] = [[[
                        sum([self.p[j-1][k][r][res] * self.w[j][i][r] for r in range(len(self.w[j][i]))]) 
                        for res in range(len(self.y[j][i]))]
                        for i in range(len(self.w[j]))] 
                        for k in range(len(self.p[j-1]))]
                else:
                    self.d[j] = 0
                
                self.e[j] = [[[
                    0 if (n==0) else
                    [[[
                        self.p[j-1][k][l][res] * self.z[j][i][res] \
                            + sum([self.q[j][r][i][res] * self.b[j-1][k][r][n][m][l][res] for r in range(len(self.q[j]))])
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))] if (m==i) else
                     [[
                        sum([self.q[j][r][i][res] * self.b[j-1][k][r][n][m][l][res] for r in range(len(self.q[j]))])
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))]  # m!=i so no p*z term

                        for m in range(len(self.w[n]))] if (n==j) else
                    [[[
                        sum([self.q[j][r][i][res] * self.b[j-1][k][r][n][m][l][res] for r in range(len(self.q[j]))])
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))]
                        for m in range(len(self.w[n]))]     # n!=j so no p*z term
                    for n in range(len(self.w))]
                    for i in range(len(self.y[j]))]
                    if ((k in self.calculate_1st_list) or (k in self.calculate_2nd_list)) else 0
                    for k in range(len(self.y[0]))]

                if ((len(self.calculate_1st_list) > 0) or (len(self.calculate_2nd_list) > 0)):
                    self.f[j] = [[
                        0 if (n==0) else
                        [[[
                            self.y[j][i][res] * self.v[j][i][n][m][l][res]
                            for res in range(len(self.y[j][i]))]
                            for l in range(len(self.w[n][m]))]
                            for m in range(len(self.w[n]))]
                        for n in range(len(self.w))]
                        for i in range(len(self.y[j]))]
                else:
                    self.f[j] = 0

                self.b[j] = [[[
                    0 if (n==0) else
                    [[[ 
                        -2 * self.f[j][i][n][m][l][res] * self.d[j][k][i][res] + self.e[j][k][i][n][m][l][res]
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))]
                        for m in range(len(self.w[n]))]
                    for n in range(len(self.w))]
                    for i in range(len(self.y[j]))]
                    if ((k in self.calculate_1st_list) or (k in self.calculate_2nd_list)) else 0            # b is used in the calculation of e, which is used in c, so b is also needed if c is needed, i.e. for 2nd derivative
                    for k in range(len(self.y[0]))]

                self.c[j] = [[[
                    0 if (n==0) else
                    [
                     [[ 
                        (4 - 6 * self.z[j][i][res]) * self.v[j][i][n][m][l][res] * self.d[j][k][i][res]**2 \
                        - 4 * self.y[j][i][res] * self.d[j][k][i][res] * self.e[j][k][i][n][m][l][res] \
                        - 2 * self.f[j][i][n][m][l][res] * sum([self.w[j][i][r] * self.a[j-1][k][r][res] for r in range(len(self.w[j][i]))]) \
                        + self.z[j][i][res] * self.a[j-1][k][l][res] \
                        + sum([self.q[j][r][i][res] * self.c[j-1][k][r][n][m][l][res] for r in range(len(self.q[j]))])
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))] if (m==i) else
                     [[ 
                        (4 - 6 * self.z[j][i][res]) * self.v[j][i][n][m][l][res] * self.d[j][k][i][res]**2 \
                        - 4 * self.y[j][i][res] * self.d[j][k][i][res] * self.e[j][k][i][n][m][l][res] \
                        - 2 * self.f[j][i][n][m][l][res] * sum([self.w[j][i][r] * self.a[j-1][k][r][res] for r in range(len(self.w[j][i]))]) \
                        + sum([self.q[j][r][i][res] * self.c[j-1][k][r][n][m][l][res] for r in range(len(self.q[j]))])
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))]  # m != i so on z*a term.

                        for m in range(len(self.w[n]))] if (n==j) else
                    [[[ 
                        (4 - 6 * self.z[j][i][res]) * self.v[j][i][n][m][l][res] * self.d[j][k][i][res]**2 \
                        - 4 * self.y[j][i][res] * self.d[j][k][i][res] * self.e[j][k][i][n][m][l][res] \
                        - 2 * self.f[j][i][n][m][l][res] * sum([self.w[j][i][r] * self.a[j-1][k][r][res] for r in range(len(self.w[j][i]))]) \
                        + sum([self.q[j][r][i][res] * self.c[j-1][k][r][n][m][l][res] for r in range(len(self.q[j]))])
                        for res in range(len(self.y[j][i]))]
                        for l in range(len(self.w[n][m]))]
                        for m in range(len(self.w[n]))]     # n != j so on z*a term.

                    for n in range(len(self.w))]
                    for i in range(len(self.y[j]))]
                    if k in self.calculate_2nd_list else 0
                    for k in range(len(self.y[0]))]
                
        '''
        self.w = np.random.random(size=(self.inputs, self.nodes))
        self.x = np.zeros(shape=self.nodes)
        self.y = np.zeros(shape=self.nodes)
        self.z = np.zeros(shape=self.nodes)
        self.q = np.zeros(size=(self.inputs, self.nodes))
        self.z = -self.y**2 - 1
        '''

    
