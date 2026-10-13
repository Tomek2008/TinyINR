from torch.nn import Module, Sequential, Parameter
import torch
import numpy as np


class FourierKANLayer(Module):
    def __init__(self, in_features, out_features, grid_size=8, input_scale=1.0, addbias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.addbias = addbias
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))

        k_decay = 1.0 / torch.arange(1, grid_size + 1, dtype=torch.float32)
        k_decay = k_decay.reshape(1, 1, 1, grid_size)
        self.fourier_coeffs = Parameter(
            torch.randn(2, out_features, in_features, grid_size)
            / (np.sqrt(in_features) * np.sqrt(grid_size))
            * k_decay
        )
        if addbias:
            self.bias = Parameter(torch.zeros(1, out_features))

        self.register_buffer(
            "harmonics", torch.arange(1, grid_size + 1, dtype=torch.float32).reshape(1, 1, grid_size)
        )

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        angles = self.input_scale * x.unsqueeze(-1) * self.harmonics
        cos_terms = torch.cos(angles)
        sin_terms = torch.sin(angles)

        y = torch.einsum("nik,oik->no", cos_terms, self.fourier_coeffs[0])
        y = y + torch.einsum("nik,oik->no", sin_terms, self.fourier_coeffs[1])

        if self.addbias:
            y = y + self.bias

        return y.reshape(*orig_shape[:-1], self.out_features)


class SineLayer(Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.register_buffer("omega_0", torch.tensor(float(omega_0)))
        self.is_first = is_first
        self.linear = torch.nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class KAN(Sequential):
    def __init__(
        self,
        in_features=3,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        num_fourier_layers=1,
        grid_size=8,
        input_scale=3.0,
        omega_0=30.0,
        addbias=True,
    ):
        total_pre_final_layers = 1 + hidden_layers
        layers = []

        for i in range(num_fourier_layers):
            in_f = in_features if i == 0 else hidden_features
            scale = input_scale if i == 0 else 1.0
            layers.append(
                FourierKANLayer(
                    in_f,
                    hidden_features,
                    grid_size,
                    input_scale=scale,
                    addbias=addbias,
                )
            )

        for i in range(num_fourier_layers, total_pre_final_layers):
            if i == 0:
                layers.append(
                    SineLayer(
                        in_features,
                        hidden_features,
                        is_first=True,
                        omega_0=omega_0,
                    )
                )
            else:
                layers.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        omega_0=omega_0,
                    )
                )

        final_layer = torch.nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_features) / omega_0
            final_layer.weight.uniform_(-bound, bound)
        layers.append(final_layer)

        super().__init__(*layers)
