import numpy as np

class SimpleCNN:
    def __init__(self):
        self.conv1_w = np.random.randn(8, 3, 3, 3) * 0.01
        self.fc_w = np.random.randn(8*37*37, 4) * 0.01

    def relu(self, x): return np.maximum(0, x)
    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x):
        z = self.conv2d(x, self.conv1_w)
        a = self.relu(z)
        out = a.reshape(a.shape[0], -1) @ self.fc_w
        return self.softmax(out)