# Cancelled--Unnamed_PINN_Library
Cancelled project to develop a new library for PINNs.

The derivative of the output of a node, with respect to any trainable or input variable, can be expressed through matrix multiplication. For activation functions whose derivatives can be expressed as a function of known quantities (e.g. y(x) = tanh(x), y'(x) = (1-y<sup>2</sup>(x))), those matrices are calculable from the known quantities. Calculating these multiplications during model evaluation and carrying the results forward enables calculations of the derivatvies in a potentially new way. This may be (a) how TensorFlow or PyTorch operate today, (b) better than how they operate, or (c) worse than how they operate. 

## Findings
- This is just exactly "auto-differentiation by forward propagaion"; I'm many years late in "discovering" this. I ended the project when I learned this.
- Expressing the calculations as matrix operations is not advantageous; list-based implementation is superior for both speed and memory.

## Contents
The value in this repository is not in the partially-drafted Python code, but in the "Expressions of Derivatives" documents. The three versions are as follows:
- (v1) is the original (computerizded version of the) derivation of the matrix representation of the derivatives.
- v2 is the algebra for the 1st and 2nd derivatives and their derivatives with respect to the weightings. v1 showed these could be written, did not write them.
- v3 is the results from v2 written with different orders of indices, as would be preferential for programming.
