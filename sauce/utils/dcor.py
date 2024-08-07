import torch
from sauce.utils.distance_matrix import euclidean_distance_matrix3


class DistanceCorrelationClassic(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) < 3 and len(y.shape) < 3, "X and Y must be 2D or 1D tensors."
        x = x if len(x.shape) == 2 else x.unsqueeze(1)
        y = y if len(y.shape) == 2 else y.unsqueeze(1)
        matrix_a = torch.sqrt(torch.sum(torch.square(x.unsqueeze(0) - x.unsqueeze(1)), dim = -1) + 1e-12)
        matrix_b = torch.sqrt(torch.sum(torch.square(y.unsqueeze(0) - y.unsqueeze(1)), dim = -1) + 1e-12)

        matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
        matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

        Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

        correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r


class DistanceCorrelation(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) < 3 and len(y.shape) < 3, "X and Y must be 2D or 1D tensors."
        x = x if len(x.shape) == 2 else x.unsqueeze(1)
        y = y if len(y.shape) == 2 else y.unsqueeze(1)
        matrix_a = euclidean_distance_matrix3(x)
        matrix_b = euclidean_distance_matrix3(y)

        matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
        matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

        Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

        correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r
