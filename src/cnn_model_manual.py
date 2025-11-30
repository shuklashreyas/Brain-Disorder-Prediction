import numpy as np
import cv2
import os
from tqdm import tqdm

IMG_SIZE = 64

def load_dataset(path):
    X, y = [], []
    class_names = sorted(os.listdir(path))

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    return np.array(X)/255.0, np.array(y), class_names


class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def forward(self, X):
        self.last_input = X
        h, w = X.shape
        fh = self.filter_size
        out_h = h - fh + 1
        out_w = w - fh + 1
        output = np.zeros((self.num_filters, out_h, out_w))

        for f in range(self.num_filters):
            filt = self.filters[f]
            for i in range(out_h):
                for j in range(out_w):
                    region = X[i:i+fh, j:j+fh]
                    output[f, i, j] = np.sum(region * filt)

        return output

    def backward(self, dL_dout, lr=1e-3):
        d_filters = np.zeros(self.filters.shape)
        fh = self.filter_size

        for f in range(self.num_filters):
            for i in range(dL_dout.shape[1]):
                for j in range(dL_dout.shape[2]):
                    region = self.last_input[i:i+fh, j:j+fh]
                    d_filters[f] += dL_dout[f, i, j] * region

        self.filters -= lr * d_filters


class ReLU:
    def forward(self, X):
        self.last_input = X
        return np.maximum(0, X)

    def backward(self, grad):
        grad[self.last_input <= 0] = 0
        return grad


class MaxPool2:
    def forward(self, X):
        self.last_input = X
        h, w = X.shape
        out = np.zeros((h//2, w//2))

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                out[i//2, j//2] = np.max(X[i:i+2, j:j+2])

        return out

    def backward(self, grad):
        dX = np.zeros(self.last_input.shape)
        h, w = grad.shape

        for i in range(h):
            for j in range(w):
                region = self.last_input[i*2:i*2+2, j*2:j*2+2]
                idx = np.unravel_index(np.argmax(region), (2,2))
                dX[i*2+idx[0], j*2+idx[1]] = grad[i,j]

        return dX


class Dense:
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_input, n_output) / n_input
        self.b = np.zeros(n_output)

    def forward(self, X):
        self.last_input = X
        return X @ self.W + self.b

    def backward(self, grad, lr=1e-3):
        dW = np.outer(self.last_input, grad)
        db = grad
        dX = grad @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db

        return dX


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9)


conv = Conv2D(8, 3)
relu = ReLU()
pool = MaxPool2()
fc = None 


def train_model(X, y, class_names, epochs=2):
    global fc
    fc = Dense((IMG_SIZE//2)*(IMG_SIZE//2)*8, len(class_names))

    for epoch in range(epochs):
        loss, correct = 0, 0

        for i in tqdm(range(len(X))):
            x = X[i].reshape(IMG_SIZE, IMG_SIZE)
            label = y[i]

            # Forward pass
            out = conv.forward(x)
            out = relu.forward(out)
            out = pool.forward(out)

            out_flat = out.flatten()
            logits = fc.forward(out_flat)
            probs = softmax(logits)

            loss += cross_entropy(probs, label)
            correct += (np.argmax(probs) == label)

            # Backward
            grad = probs
            grad[label] -= 1

            grad = fc.backward(grad)
            grad = grad.reshape(out.shape)
            grad = pool.backward(grad)
            grad = relu.backward(grad)
            conv.backward(grad)

        print(f"Epoch {epoch+1} | Loss: {loss/len(X):.4f} | Acc: {correct/len(X):.4f}")


if __name__ == "__main__":
    train_path = "data/CombinedDataset/train"
    test_path = "data/CombinedDataset/test"

    X_train, y_train, class_names = load_dataset(train_path)
    X_test, y_test, _ = load_dataset(test_path)

    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    train_model(X_train, y_train, class_names, epochs=2)
