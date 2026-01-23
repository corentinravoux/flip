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
        """Simple fully connected regression network.

        Args:
            input_dimension: Number of input features.
            dimension_hidden_layers: Width of hidden layers.
            number_hidden_layers: Number of hidden layers.
            output_dimension: Number of output targets.
            activation_function: Torch activation class to use.
        """
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
        """Forward pass returning network output."""
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
    """Train a torch model on normalized data.

    Args:
        number_epochs: Number of training epochs.
        model: Torch model to train.
        normalized_input: Normalized input array ``(n_samples, n_features)``.
        normalized_output: Normalized target array ``(n_samples, n_targets)``.
        optimizer: Torch optimizer instance.
        loss_function: Torch loss function.
        verbose: If True, logs periodic losses.
        model_name: Label used in logging.
    """
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
    """Train neural network emulators for covariance matrices.

    Normalizes inputs and outputs; trains separate nets per covariance term.
    For square covariances (gg/vv) also trains a variance net.

    Args:
        square_covariance: Whether the covariance is square (gg/vv) or rectangular (gv).
        output_variance: Array ``(n_terms, n_samples)`` diagonal entries; ignored if not square.
        output_non_diagonal: Array ``(n_terms, n_samples, n_nd)`` flattened non-diagonals.
        parameter_values: Array ``(n_samples, n_params)`` emulator inputs.
        verbose: If True, prints training progress.
        dimension_hidden_layers: Hidden layer width.
        number_hidden_layers: Hidden layer count.
        number_epochs: Epochs to train.
        activation_function: Torch activation class.
        loss_function: Torch loss function.
        tolerance_optimizer: Learning rate for Adam optimizer.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        Tuple of lists/dicts: ``(nn_models_variance, nn_models_non_diagonal,
        nn_evaluation_dictionary_variance, nn_evaluation_dictionary_non_diagonal)``.
    """

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
    """Evaluate a trained NN emulator, denormalizing output.

    Args:
        model: Trained torch model.
        evaluation_value: Array ``(1, n_params)`` input values.
        evaluation_dictionary: Dict with normalization stats (input_mean/std, output_mean/std).

    Returns:
        Tuple ``(output, None)`` where output is the denormalized numpy array.
    """

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
