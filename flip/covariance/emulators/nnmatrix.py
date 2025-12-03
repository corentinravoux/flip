import numpy as np

from flip.utils import create_log

log = create_log()

try:
    import torch

    torch_install = True

except ImportError:
    torch_install = False
    log.add(
        "Install pytorch to use the nnmatrix emulator",
        level="warning",
    )
if torch_install:
    default_regression_object = torch.nn.Module
    default_activation_function = torch.nn.ReLU
    default_loss_function = torch.nn.MSELoss()
else:
    default_regression_object = object
    default_activation_function = None
    default_loss_function = None

_emulator_type = "matrix"


class RegressionNet(default_regression_object):
    def __init__(
        self,
        input_dimension,
        dimension_hidden_layers=64,
        number_hidden_layers=3,
        output_dimension=1,
        activation_function=default_activation_function,
    ):
        super().__init__()
        layers = []

        layers.append(torch.nn.Linear(input_dimension, dimension_hidden_layers))
        layers.append(activation_function())

        for _ in range(number_hidden_layers - 1):
            layers.append(
                torch.nn.Linear(dimension_hidden_layers, dimension_hidden_layers)
            )
            layers.append(activation_function())

        layers.append(torch.nn.Linear(dimension_hidden_layers, output_dimension))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_torch_model(
    number_epochs,
    model,
    normalized_input,
    normalized_output,
    optimizer,
    loss_function,
    verbose,
    model_name,
):
    normalized_input = torch.tensor(normalized_input, dtype=torch.float32)
    normalized_output = torch.tensor(normalized_output, dtype=torch.float32)
    for epoch in range(1, number_epochs + 1):
        prediction = model(normalized_input)
        loss = loss_function(prediction, normalized_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            if epoch == 1 or epoch % 500 == 0:
                log.add(
                    f"Epoch {epoch:4d} for model {model_name} | loss = {loss.item():.6f}"
                )


def train(
    square_covariance,
    output_variance,
    output_non_diagonal,
    parameter_values,
    verbose=False,
    dimension_hidden_layers=64,
    number_hidden_layers=3,
    number_epochs=3000,
    activation_function=default_activation_function,
    loss_function=default_loss_function,
    tolerance_optimizer=1e-3,
    **kwargs,
):

    parameter_values_mean, parameter_values_std = (
        parameter_values.mean(axis=0),
        parameter_values.std(axis=0),
    )
    normalized_parameter_values = (
        parameter_values - parameter_values_mean
    ) / parameter_values_std

    if square_covariance:
        nn_models_variance = []
        nn_evaluation_dictionary_variance = [
            {
                "input_mean": parameter_values_mean,
                "input_std": parameter_values_std,
                "output_mean": np.mean(output_variance[j]),
                "output_std": np.std(output_variance[j]),
            }
            for j in range(len(output_variance))
        ]

    else:
        nn_models_variance = None
        nn_evaluation_dictionary_variance = None

    nn_models_non_diagonal = []

    nn_evaluation_dictionary_non_diagonal = [
        {
            "input_mean": parameter_values_mean,
            "input_std": parameter_values_std,
            "output_mean": np.mean(output_non_diagonal[j]),
            "output_std": np.std(output_non_diagonal[j]),
        }
        for j in range(len(output_non_diagonal))
    ]

    for j in range(len(output_non_diagonal)):
        if square_covariance:
            normalized_nn_output_variance = (
                output_variance[j][:, np.newaxis]
                - nn_evaluation_dictionary_variance[j]["output_mean"]
            ) / nn_evaluation_dictionary_variance[j]["output_std"]

            model_variance = RegressionNet(
                input_dimension=normalized_parameter_values.shape[1],
                dimension_hidden_layers=dimension_hidden_layers,
                number_hidden_layers=number_hidden_layers,
                output_dimension=normalized_nn_output_variance.shape[1],
                activation_function=activation_function,
            )
            optimizer = torch.optim.Adam(
                model_variance.parameters(), lr=tolerance_optimizer
            )

            train_torch_model(
                number_epochs,
                model_variance,
                normalized_parameter_values,
                normalized_nn_output_variance,
                optimizer,
                loss_function,
                verbose,
                "variance",
            )

            nn_models_variance.append(model_variance)

        normalized_nn_output_non_diagonal = (
            output_non_diagonal[j]
            - nn_evaluation_dictionary_non_diagonal[j]["output_mean"]
        ) / nn_evaluation_dictionary_non_diagonal[j]["output_std"]
        model_non_diagonal = RegressionNet(
            input_dimension=normalized_parameter_values.shape[1],
            dimension_hidden_layers=dimension_hidden_layers,
            number_hidden_layers=number_hidden_layers,
            output_dimension=normalized_nn_output_non_diagonal.shape[1],
            activation_function=activation_function,
        )
        optimizer = torch.optim.Adam(
            model_non_diagonal.parameters(), lr=tolerance_optimizer
        )

        train_torch_model(
            number_epochs,
            model_non_diagonal,
            normalized_parameter_values,
            normalized_nn_output_non_diagonal,
            optimizer,
            loss_function,
            verbose,
            "non_diagonal",
        )

        nn_models_non_diagonal.append(model_non_diagonal)

    return (
        nn_models_variance,
        nn_models_non_diagonal,
        nn_evaluation_dictionary_variance,
        nn_evaluation_dictionary_non_diagonal,
    )


def evaluate(
    model,
    evaluation_value,
    evaluation_dictionary,
):

    normalized_evaluation_value = (
        evaluation_value - evaluation_dictionary["input_mean"]
    ) / evaluation_dictionary["input_std"]

    normalized_output = model(
        torch.tensor(normalized_evaluation_value, dtype=torch.float32)
    )
    numpy_normalized_output = normalized_output.detach().cpu().numpy()
    output = (
        numpy_normalized_output * evaluation_dictionary["output_std"]
        + evaluation_dictionary["output_mean"]
    )
    return (output, None)
