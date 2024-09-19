import numpy as np
#Name:   Skand Vijay
#AndrewID: skandv

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape  # TODO
        se = (A - Y) ** 2  # TODO
        sse = np.sum(se)  # TODO
        mse = sse / (self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = (2 / (self.N * self.C)) * (self.A - self.Y)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N, C = A.shape  # TODO

        Ones_C = np.ones((1, C), dtype='f')  # TODO
        Ones_N = np.ones((N, 1), dtype='f')  # TODO

        expA = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.softmax = expA / np.dot(np.sum(expA, axis=1, keepdims=True), Ones_C)  # TODO
        crossentropy = -np.dot(Y * np.log(self.softmax + 1e-12), Ones_C.T)  # TODO
        sum_crossentropy = np.dot(Ones_N.T, crossentropy)   # TODO
        L = sum_crossentropy / N

        return L[0]

    def backward(self):
        N, C = self.A.shape 
        dLdA = (self.softmax - self.Y) / N   # TODO

        return dLdA
