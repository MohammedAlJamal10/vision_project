import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.cnn.layers import (
    Conv2D,
    ReLU,
    MaxPool2D,
    Flatten,
    Linear,
)

from models.cnn.loss import SoftmaxCrossEntropyLoss


class SimpleCNN:

    def __init__(
        self,
        in_channels=3,
        n_classes=6,
        img_size=96,
        lambda_l2=1e-4,
    ):

        self.lambda_l2 = lambda_l2
        self.n_classes = n_classes

        # =========================
        # BLOCK 1
        # =========================

        self.conv1 = Conv2D(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2)

        # =========================
        # BLOCK 2
        # =========================

        self.conv2 = Conv2D(32, 64, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2)

        # =========================
        # BLOCK 3
        # =========================

        self.conv3 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D(pool_size=2)

        self.flatten = Flatten()

        flat_h = img_size // 8
        flat_dim = 128 * flat_h * flat_h

        # =========================
        # FC
        # =========================

        self.fc1 = Linear(flat_dim, 256)
        self.relu4 = ReLU()

        self.fc2 = Linear(256, n_classes)

        self.loss_fn = SoftmaxCrossEntropyLoss()

        self._param_layers = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.fc1,
            self.fc2,
        ]

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, x, y=None):

        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.conv3.forward(out)
        out = self.relu3.forward(out)
        out = self.pool3.forward(out)

        out = self.flatten.forward(out)

        out = self.fc1.forward(out)
        out = self.relu4.forward(out)

        logits = self.fc2.forward(out)

        if y is None:
            return logits, None

        loss = self.loss_fn.forward(logits, y)

        for layer in self._param_layers:
            loss += (self.lambda_l2 / 2) * np.sum(layer.W ** 2)

        return logits, loss

    # =========================================================
    # BACKWARD
    # =========================================================

    def backward(self):

        dout = self.loss_fn.backward()

        dout = self.fc2.backward(dout)

        dout = self.relu4.backward(dout)
        dout = self.fc1.backward(dout)

        dout = self.flatten.backward(dout)

        dout = self.pool3.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)

        # L2 REGULARIZATION

        for layer in self._param_layers:
            layer.dW += self.lambda_l2 * layer.W

    # =========================================================
    # TRAINING
    # =========================================================

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer,
        batch_size=32,
        max_epochs=100,
        grad_clip_norm=5.0,
        scheduler=None,
        augment_fn=None,
        patience=10,
        seed=42,
    ):

        rng = np.random.default_rng(seed)

        N = len(y_train)

        best_val_acc = -1
        best_params = None
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):

            perm = rng.permutation(N)

            X_sh = X_train[perm]
            y_sh = y_train[perm]

            epoch_loss = 0
            n_batches = 0

            # =========================================
            # TRAIN LOOP
            # =========================================

            for start in range(0, N, batch_size):

                end = start + batch_size

                Xb = X_sh[start:end]
                yb = y_sh[start:end]

                if augment_fn is not None:
                    Xb = augment_fn(Xb, rng)

                _, loss = self.forward(Xb, yb)

                self.backward()

                # =========================================
                # GRAD CLIP
                # =========================================

                grads = []

                for layer in self._param_layers:
                    grads.append(layer.dW.ravel())
                    grads.append(layer.db.ravel())

                grads = np.concatenate(grads)

                grad_norm = np.linalg.norm(grads)

                if grad_norm > grad_clip_norm:

                    scale = grad_clip_norm / grad_norm

                    for layer in self._param_layers:
                        layer.dW *= scale
                        layer.db *= scale

                # =========================================
                # OPTIMIZER STEP
                # =========================================

                for idx, layer in enumerate(self._param_layers):

                    layer.W, layer.b = optimizer.step(
                        layer.W,
                        layer.b,
                        layer.dW,
                        layer.db,
                        key=id(layer),
                        update_t=(idx == 0),
                    )

                epoch_loss += loss
                n_batches += 1

            train_loss = epoch_loss / n_batches

            val_loss = self._eval_loss(X_val, y_val)

            train_acc = self.score(X_train, y_train)

            val_acc = self.score(X_val, y_val)

            current_lr = optimizer.lr

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            if epoch == 1 or epoch % 5 == 0:

                print(
                    f"  Epoch {epoch:3d}/{max_epochs} | "
                    f"tr_loss={train_loss:.4f} | "
                    f"va_loss={val_loss:.4f} | "
                    f"tr_acc={train_acc:.4f} | "
                    f"va_acc={val_acc:.4f} | "
                    f"lr={current_lr:.2e}"
                )

            # =========================================
            # EARLY STOPPING
            # =========================================

            if val_acc > best_val_acc:

                best_val_acc = val_acc
                best_params = self.get_params()

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= patience:

                print(
                    f"  Early stopping: val_acc did not improve "
                    f"for {patience} epochs."
                )

                if best_params is not None:
                    self.set_params(best_params)

                break

            # =========================================
            # LR SCHEDULER
            # =========================================

            if scheduler is not None:
                optimizer.lr = scheduler.step(epoch, val_loss)

        return self

    # =========================================================
    # INFERENCE
    # =========================================================

    def _forward_inference(self, x):

        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.conv3.forward(out)
        out = self.relu3.forward(out)
        out = self.pool3.forward(out)

        out = self.flatten.forward(out)

        out = self.fc1.forward(out)
        out = self.relu4.forward(out)

        logits = self.fc2.forward(out)

        return logits

    def predict(self, X, batch_size=64):

        preds = []

        for start in range(0, len(X), batch_size):

            Xb = X[start:start + batch_size]

            logits = self._forward_inference(Xb)

            pred = np.argmax(logits, axis=1)

            preds.append(pred)

        return np.concatenate(preds)

    def predict_proba(self, X, batch_size=64):

        probs = []

        for start in range(0, len(X), batch_size):

            Xb = X[start:start + batch_size]

            logits = self._forward_inference(Xb)

            logits = logits - logits.max(axis=1, keepdims=True)

            exp = np.exp(logits)

            prob = exp / exp.sum(axis=1, keepdims=True)

            probs.append(prob)

        return np.concatenate(probs)

    def score(self, X, y):

        preds = self.predict(X)

        return np.mean(preds == y)

    def _eval_loss(self, X, y, batch_size=64):

        total_loss = 0

        for start in range(0, len(X), batch_size):

            Xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]

            logits = self._forward_inference(Xb)

            z = logits - logits.max(axis=1, keepdims=True)

            exp = np.exp(z)

            probs = exp / exp.sum(axis=1, keepdims=True)

            p = probs[np.arange(len(yb)), yb]

            total_loss += -np.sum(np.log(p + 1e-9))

        return total_loss / len(X)

    # =========================================================
    # SAVE / LOAD
    # =========================================================

    def get_params(self):

        return {

            "conv1_W": self.conv1.W.copy(),
            "conv1_b": self.conv1.b.copy(),

            "conv2_W": self.conv2.W.copy(),
            "conv2_b": self.conv2.b.copy(),

            "conv3_W": self.conv3.W.copy(),
            "conv3_b": self.conv3.b.copy(),

            "fc1_W": self.fc1.W.copy(),
            "fc1_b": self.fc1.b.copy(),

            "fc2_W": self.fc2.W.copy(),
            "fc2_b": self.fc2.b.copy(),
        }

    def set_params(self, params):

        self.conv1.W = params["conv1_W"]
        self.conv1.b = params["conv1_b"]

        self.conv2.W = params["conv2_W"]
        self.conv2.b = params["conv2_b"]

        self.conv3.W = params["conv3_W"]
        self.conv3.b = params["conv3_b"]

        self.fc1.W = params["fc1_W"]
        self.fc1.b = params["fc1_b"]

        self.fc2.W = params["fc2_W"]
        self.fc2.b = params["fc2_b"]