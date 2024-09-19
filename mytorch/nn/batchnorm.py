import numpy as np
#Name:   Skand Vijay
#AndrewID: skandv
class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]  # TODO
        self.M = np.mean(Z, axis=0, keepdims=True)  # TODO
        self.V = np.var(Z, axis=0, keepdims=True)  # TODO

        if eval == False:
            # training mode
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)  # TODO
            self.BZ = self.BW * self.NZ + self.Bb  # TODO

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M  # TODO
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V  # TODO
            
            return self.BZ
        else:
            # inference mode
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)  # TODO
            BZ = self.BW * NZ + self.Bb  # TODO

        return BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW

        dLdV = np.sum(
            dLdNZ * (self.Z - self.M) * -0.5 * (self.V + self.eps) ** -1.5,
            axis=0,
            keepdims=True,
        )
        dLdM = np.sum(
            dLdNZ * -1 / np.sqrt(self.V + self.eps), axis=0, keepdims=True
        ) + dLdV * np.mean(-2 * (self.Z - self.M), axis=0, keepdims=True)

        dLdZ = (
            dLdNZ / np.sqrt(self.V + self.eps)
            + dLdV * 2 * (self.Z - self.M) / self.N
            + dLdM / self.N
        )

        return dLdZ
