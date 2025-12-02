import numpy as np
import cv2
import os
from tqdm import tqdm

np.random.seed(42)


IMG_SIZE   = 64
TRAIN_PATH = "data/CombinedDataset/train"
TEST_PATH  = "data/CombinedDataset/test"

SAVE_DIR   = "MLModels"
SAVE_PATH  = os.path.join(SAVE_DIR, "manual_cnn_v2.npz")


def load_dataset(path):
    X, y = [], []

    class_names = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])

    print("Classes:", class_names)

    for label, cls in enumerate(class_names):
        folder = os.path.join(path, cls)
        imgs = os.listdir(folder)

        for img_name in imgs:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)

    print(f"Loaded {len(X)} images from {path}.")
    return X, y, class_names


def train_val_split(X, y, val_ratio=0.2):
    N = len(X)
    idx = np.random.permutation(N)
    split = int(N * (1 - val_ratio))

    train_idx = idx[:split]
    val_idx   = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    return X_train, y_train, X_val, y_val


class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # simple He-ish init
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def forward(self, X):
        # X: (H, W)
        self.last_input = X
        h, w = X.shape
        fh = self.filter_size

        out_h = h - fh + 1
        out_w = w - fh + 1
        out = np.zeros((self.num_filters, out_h, out_w), dtype=np.float32)

        for f in range(self.num_filters):
            filt = self.filters[f]
            for i in range(out_h):
                for j in range(out_w):
                    region = X[i:i+fh, j:j+fh]
                    out[f, i, j] = np.sum(region * filt)

        return out

    def backward(self, d_out, lr=1e-3):
        d_filters = np.zeros_like(self.filters)
        fh = self.filter_size
        _, out_h, out_w = d_out.shape

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = self.last_input[i:i+fh, j:j+fh]
                    d_filters[f] += d_out[f, i, j] * region

        self.filters -= lr * d_filters


class ReLU:
    def forward(self, X):
        self.last_input = X
        return np.maximum(0, X)

    def backward(self, grad):
        grad = grad.copy()
        grad[self.last_input <= 0] = 0
        return grad


class MaxPool2:
    """
    2x2 max pool with stride 2.
    Input:  (C, H, W)
    Output: (C, H/2, W/2)
    """
    def forward(self, X):
        self.last_input = X
        C, H, W = X.shape
        out = np.zeros((C, H // 2, W // 2), dtype=np.float32)

        for c in range(C):
            for i in range(0, H, 2):
                for j in range(0, W, 2):
                    region = X[c, i:i+2, j:j+2]
                    out[c, i//2, j//2] = np.max(region)

        return out

    def backward(self, grad):
        C, H_half, W_half = grad.shape
        dX = np.zeros_like(self.last_input)

        for c in range(C):
            for i in range(H_half):
                for j in range(W_half):
                    region = self.last_input[c, i*2:(i+1)*2, j*2:(j+1)*2]
                    idx = np.unravel_index(np.argmax(region), (2, 2))
                    dX[c, i*2 + idx[0], j*2 + idx[1]] = grad[c, i, j]

        return dX


class Dense:
    def __init__(self, n_input, n_output):
        self.W = np.random.randn(n_input, n_output) / np.sqrt(n_input)
        self.b = np.zeros(n_output, dtype=np.float32)

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
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9)


def forward_pass(img, conv, relu, pool, fc):
    """
    img: (IMG_SIZE, IMG_SIZE), grayscale in [0,1]
    returns: (conv_out, relu_out, pool_out, flat, logits, probs)
    """
    conv_out = conv.forward(img)
    relu_out = relu.forward(conv_out)
    pool_out = pool.forward(relu_out)

    flat = pool_out.flatten()
    logits = fc.forward(flat)
    probs = softmax(logits)

    return conv_out, relu_out, pool_out, flat, logits, probs


def evaluate(X, y, conv, relu, pool, fc):
    N = len(X)
    total_loss = 0.0
    correct = 0

    for i in range(N):
        img = X[i].reshape(IMG_SIZE, IMG_SIZE)
        label = y[i]
        _, _, _, _, _, probs = forward_pass(img, conv, relu, pool, fc)
        total_loss += cross_entropy(probs, label)
        correct += (np.argmax(probs) == label)

    return total_loss / N, correct / N


def train_manual(
    X_train,
    y_train,
    X_val,
    y_val,
    class_names,
    epochs=2,
    batch_size=8,
    lr=1e-3,
    save_path=None,
):
    num_classes = len(class_names)

    flat_size = ((IMG_SIZE - 2) // 2) * ((IMG_SIZE - 2) // 2) * 8
    print("Flat size:", flat_size, "| Num classes:", num_classes)

    conv = Conv2D(8, 3)
    relu = ReLU()
    pool = MaxPool2()
    fc   = Dense(flat_size, num_classes)

    N = len(X_train)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_train = X_train[perm]
        y_train = y_train[perm]

        total_loss = 0.0
        correct = 0

        num_batches = N // batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}")
        for b in pbar:
            xb = X_train[b * batch_size:(b + 1) * batch_size]
            yb = y_train[b * batch_size:(b + 1) * batch_size]

            for i in range(len(xb)):
                img = xb[i].reshape(IMG_SIZE, IMG_SIZE)
                label = yb[i]

                # Forward
                conv_out, relu_out, pool_out, flat, logits, probs = forward_pass(
                    img, conv, relu, pool, fc
                )

                total_loss += cross_entropy(probs, label)
                correct += (np.argmax(probs) == label)

                # Backward
                grad_logits = probs
                grad_logits[label] -= 1  # dL/dlogits

                grad_flat = fc.backward(grad_logits, lr=lr)
                grad_pool = grad_flat.reshape(pool_out.shape)
                grad_relu = pool.backward(grad_pool)
                grad_conv = relu.backward(grad_relu)
                conv.backward(grad_conv, lr=lr)

            pbar.set_postfix({"loss": total_loss / ((b + 1) * batch_size)})

        train_loss = total_loss / N
        train_acc  = correct / N

        val_loss, val_acc = evaluate(X_val, y_val, conv, relu, pool, fc)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            conv_filters=conv.filters,
            fc_W=fc.W,
            fc_b=fc.b,
            class_names=np.array(class_names),
        )
        print(f"Saved manual CNN weights to: {save_path}")

    return conv, relu, pool, fc


if __name__ == "__main__":
    X_all, y_all, class_names = load_dataset(TRAIN_PATH)
    X_test, y_test, _         = load_dataset(TEST_PATH)

    X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, val_ratio=0.2)

    conv, relu, pool, fc = train_manual(
        X_train,
        y_train,
        X_val,
        y_val,
        class_names,
        epochs=5,    
        batch_size=8,
        lr=1e-3,
        save_path=SAVE_PATH,
    )

    test_loss, test_acc = evaluate(X_test, y_test, conv, relu, pool, fc)
    print(f"Manual CNN test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}")
