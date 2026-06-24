# nightlanp/wrappers.py
import pickle
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from knp.data import get_ragged_tensor, create_stratified_np_dataset  # noqa
from knp.models import ANP, NP, CNP


class LightCurveNeuralProcess:
    def __init__(self, model_type="ANP", latent_dim=128, learning_rate=1e-4):
        self.model_type = model_type
        self.time_scaler = MinMaxScaler()
        if model_type == "ANP":
            self.keras_model = ANP(output_dims=1, num_latents=latent_dim)
        elif model_type == "NP":
            self.keras_model = NP(output_dims=1, num_latents=latent_dim)
        elif model_type == "CNP":
            self.keras_model = CNP(output_dims=1)

        self.keras_model.compile(optimizer=keras.optimizers.Adam(learning_rate))

    def fit_population(self, lc_list, meta_df, epochs=10000):
        """
        Meta-learns from a population of light curves.
        lc_list: List of numpy arrays [time, flux, filter, flux_err]
        """
        all_times = np.concatenate([lc[:, 0] for lc in lc_list]).reshape(-1, 1)
        self.time_scaler.fit(all_times)

        X_list, y_list = [], []
        for lc in lc_list:
            t_scaled = self.time_scaler.transform(lc[:, 0].reshape(-1, 1))
            filters = lc[:, 2].reshape(-1, 1)
            flux = lc[:, 1].reshape(
                -1, 1
            )  # Assuming flux is roughly scaled bw 0 & 1000

            X_list.append(np.hstack([t_scaled, filters]))
            y_list.append(flux)

        X_ragged = get_ragged_tensor(X_list)
        y_ragged = get_ragged_tensor(y_list)

        self.keras_model.train(
            X_train=X_ragged,
            y_train=y_ragged,
            epochs=epochs,
            optimizer=self.keras_model.optimizer,
            stratify_labels=meta_df["SIM_TYPE_NAME"].to_numpy(),
        )
        print("Population training complete.")

    def predict(self, ctx_t, ctx_f, ctx_flt, target_t, target_flt):
        """
        Predicts for a single object.
        Inputs are raw physical units.
        """
        c_t_norm = self.time_scaler.transform(ctx_t.reshape(-1, 1))
        t_t_norm = self.time_scaler.transform(target_t.reshape(-1, 1))

        c_x = np.hstack([c_t_norm, ctx_flt.reshape(-1, 1)])[np.newaxis, ...]
        c_y = ctx_f.reshape(-1, 1)[np.newaxis, ...]
        t_x = np.hstack([t_t_norm, target_flt.reshape(-1, 1)])[np.newaxis, ...]

        mean, std = self.keras_model.predict(c_x, c_y, t_x)

        return mean.numpy()[0], std.numpy()[0]

    def save(self, path):
        self.keras_model.save(f"{path}.keras")
        with open(f"{path}_scaler.pkl", "wb") as f:
            pickle.dump(self.time_scaler, f)
