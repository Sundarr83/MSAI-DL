"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        # raise NotImplementedError("ClassificationLoss.forward() is not implemented")
        return F.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        
        # Define the linear layer: input size is h * w * 3 (flattened image), output size is num_classes
        input_dim = h * w * 3  # Each image has 3 channels and is h x w
        self.linear = nn.Linear(input_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input: from (B, 3, H, W) to (B, 3*H*W)
        x = x.view(x.size(0), -1)  # Flatten the image for each batch sample
        
        # Pass through the linear layer
        logits = self.linear(x)
        
        return logits


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
            hidden_dim: int, number of units in the hidden layer (default is 128)
        """
        super().__init__()

        # raise NotImplementedError("MLPClassifier.__init__() is not implemented")
        input_dim = h * w * 3  # Each image has 3 channels and is h x w (3 * 64 * 64)
        
        # Define MLP layers: input to hidden layer, then hidden to output layer
        self.mlp = nn.Sequential(
            nn.Flatten(),  # Flatten the input from (B, 3, 64, 64) to (B, 3*64*64)
            nn.Linear(input_dim, hidden_dim),  # Input to hidden layer
            nn.ReLU(),  # Non-linear activation
            nn.Linear(hidden_dim, num_classes)  # Hidden layer to output (num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # raise NotImplementedError("MLPClassifier.forward() is not implemented")
        logits = self.mlp(x)
        return logits


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128, 
        num_layers: int = 4
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layershidden_dim: int, number of units in each hidden layer (default is 128)
            num_layers: int, number of layers in the network (including input and output layers, default is 4)
        """
        super().__init__()

        input_dim = h * w * 3  # Each image has 3 channels and is h x w (3 * 64 * 64)

        # Ensure num_layers is at least 4
        assert num_layers >= 4, "num_layers should be at least 4 to build a deep model."

        # First layer (input to first hidden layer)
        layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        # Hidden layers: add (num_layers - 2) additional hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (hidden_dim to num_classes)
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Define the MLP using nn.Sequential
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")
        logits = self.mlp(x)
        return logits


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128, 
        num_layers: int = 4
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        input_dim = h * w * 3  # Each image has 3 channels and is h x w (3 * 64 * 64)
        
        # Ensure num_layers is at least 4
        assert num_layers >= 4, "num_layers should be at least 4 to build a deep model with residual connections."

        # First layer (input to first hidden layer)
        self.input_layer = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # Hidden layers with residual connections
        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.residual_layers.append(nn.Linear(hidden_dim, hidden_dim))  # Adding residual layers

        # Output layer (hidden_dim to num_classes)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Pass through the input layer
        x = self.input_layer(x)

        # Pass through the residual layers
        for layer in self.residual_layers:
            residual = x.clone()  # Clone to avoid modifying the original tensor in place
            x = layer(x)  # Apply linear transformation
            x = self.relu(x)  # Apply ReLU activation
            x = x + residual  # Add residual connection

        # Pass through the output layer
        logits = self.output_layer(x)
        return logits


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
