#################################################################
# Code      NN via MM Library utilization code
# Version   1.0
# Date      2025-09-16
# Author    Dan Humfeld, DanHumfeld@GraceECLLC.com
# Note      This library enables the training and use of NNs
#           including PINNs.
#
#################################################################
# Importing libraries 
#################################################################
import random
import math
import nnl


#################################################################
# Main Code 
#################################################################
# Use nodes = [3 2 2 1].
my_model = nnl.Model(3)
my_model.add_layer(2, 'tanh')
my_model.add_layer(2, 'tanh')
my_model.add_layer(1, 'tanh')

# Try something larger: nodes = [3 32 32 32 32 1]
#my_model = nnl.Model(3)
#my_model.add_layer(32, 'tanh')
#my_model.add_layer(32, 'tanh')
#my_model.add_layer(32, 'tanh')
#my_model.add_layer(32, 'tanh')
#my_model.add_layer(1, 'tanh')

my_model.compile(test_mode = True)
residual_list = [[0.0,1.0,2.0],
                 [0.1,1.1,2.1],
                 [0.2,1.2,2.2],
                 [0.3,1.2,2.3],
                 [0.4,1.4,2.4]]
my_model.calculate_required_data(residual_list)