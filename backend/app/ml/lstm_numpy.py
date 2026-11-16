"""
app/ml/lstm_numpy.py

The LSTM autoencoder, run with plain numpy. No TensorFlow / Keras is needed to RUN it, so it
loads on any backend, whatever TensorFlow version is installed. It behaves like the Keras
model for the two things the app uses: .predict(x) and .input_shape.

Architecture (same as the notebook):
  LSTM(units, return_sequences=False) -> RepeatVector(seq_len)
  -> LSTM(units, return_sequences=True) -> TimeDistributed(Dense(n_features))

The weights are the 8 arrays from keras_model.get_weights(), saved as lstm_autoencoder_weights.npz
by the notebook export cell (and by train_runner after a retrain).
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _lstm(x, kernel, recurrent, bias, return_sequences):
    units = recurrent.shape[0]
    batch, steps, _ = x.shape
    h = np.zeros((batch, units))
    c = np.zeros((batch, units))
    xk = x @ kernel + bias                      # Keras gate order: input, forget, cell, output
    out = np.empty((batch, steps, units)) if return_sequences else None
    for t in range(steps):
        z = xk[:, t, :] + h @ recurrent
        i = _sigmoid(z[:, :units])
        f = _sigmoid(z[:, units:2 * units])
        g = np.tanh(z[:, 2 * units:3 * units])
        o = _sigmoid(z[:, 3 * units:])
        c = f * c + i * g
        h = o * np.tanh(c)
        if return_sequences:
            out[:, t, :] = h
    return out if return_sequences else h


def weights_look_valid(w) -> bool:
    if len(w) != 8:
        return False
    k1, r1, b1, k2, r2, b2, wd, bd = w
    u = r1.shape[0]
    return (k1.shape[1] == 4 * u and r1.shape == (u, 4 * u) and b1.shape == (4 * u,)
            and k2.shape == (u, 4 * u) and r2.shape == (u, 4 * u) and b2.shape == (4 * u,)
            and wd.shape[0] == u and bd.shape == (wd.shape[1],))


class NumpyLSTM:
    def __init__(self, weights, seq_len: int):
        weights = [np.asarray(a, dtype=np.float64) for a in weights]
        if not weights_look_valid(weights):
            raise ValueError("LSTM weights do not match the expected autoencoder layout")
        self.weights = weights
        self.input_shape = (None, int(seq_len), int(weights[0].shape[0]))

    @classmethod
    def from_npz(cls, path: str, seq_len: int):
        z = np.load(path)
        return cls([z[f"w{i}"] for i in range(len(z.files))], seq_len)

    def predict(self, x, verbose=0, batch_size=2048):
        x = np.asarray(x, dtype=np.float64)
        k1, r1, b1, k2, r2, b2, wd, bd = self.weights
        outs = []
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size]
            h = _lstm(xb, k1, r1, b1, return_sequences=False)
            rep = np.repeat(h[:, None, :], xb.shape[1], axis=1)
            seq = _lstm(rep, k2, r2, b2, return_sequences=True)
            outs.append(seq @ wd + bd)
        return np.concatenate(outs, axis=0) if outs else np.zeros_like(x)