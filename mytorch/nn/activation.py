import numpy as np
import scipy

#Name:   Skand Vijay
#AndrewID: skandv

class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):
        self.A = 1/(1+np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        dLdZ = dLdA *(self.A - (self.A)*(self.A))
        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self,Z):
        self.A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        return self.A
    
    def backward(self,dLdA):
        dLdZ = dLdA*(1- (self.A * self.A))
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    
    
    
    def forward(self, Z):
        def relu(x):
            return np.maximum(0,x)
        self.A = relu(Z)
        return self.A
    
    def backward(self,dLdA):
        def relud(x):
            return np.where(x > 0, 1, 0)
        
        dLdZ = dLdA * relud(self.A)
        return dLdZ
        



class GELU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on GELU.
    """

    def forward(self,Z):
        self.Z= Z
        self.A = (0.5 * Z) * (1+scipy.special.erf(Z/np.sqrt(2)))
        return self.A

    def backward(self,dLdA):
        dLdZ = dLdA * ((0.5*((1+(scipy.special.erf((self.Z)/np.sqrt(2)))))) + ((((self.Z)/np.sqrt(2 * np.pi)))* np.exp(((-0.5)*((self.Z * self.Z))))))

        return dLdZ


class Softmax:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Softmax.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        """
        Z_m = np.max(Z, axis=1, keepdims=True)
        Z_stable = Z - Z_m
    
        # Exponentiate the stabilized values
        exp_Z = np.exp(Z_stable)
    
    # Sum of exponentials along the rows
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    
    # Compute the softmax values
        self.A = exp_Z / sum_exp_Z

        return self.A
    
    def backward(self, dLdA):
        # Get the batch size (N) and number of classes (C)
        N = self.A.shape[0]
        C = self.A.shape[1]

    # Initialize the gradient dLdZ with zeros
        dLdZ = np.zeros((N, C))
    
    # Loop over each data point in the batch
        for i in range(N):
            # Initialize the Jacobian matrix J with zeros
            J = np.zeros((C, C))

            # Fill the Jacobian matrix as per the Softmax backward equation
            for m in range(C):
                for n in range(C):
                    if m == n:
                        # Diagonal terms of the Jacobian (a_m * (1 - a_m))
                        J[m, n] = self.A[i, m] * (1 - self.A[i, m])
                    else:
                        # Off-diagonal terms (-a_m * a_n)
                        J[m, n] = -self.A[i, m] * self.A[i, n]

        # Compute the gradient with respect to the input Z for the i-th data point
            dLdZ[i, :] = J @ dLdA[i, :]

        return dLdZ
    

    