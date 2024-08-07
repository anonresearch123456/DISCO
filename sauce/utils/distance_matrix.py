import torch


def euclidean_distance_matrix(matrix: torch.Tensor):
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(1)
    return (matrix.unsqueeze(1) - matrix.unsqueeze(0)).pow(2).sum(2).clamp(min=0.0).sqrt()


def euclidean_distance_matrix2(matrix: torch.Tensor):
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(1)
    GX = torch.matmul(matrix, matrix.T)
    KX_I = (torch.diag(GX) - GX)
    KX = KX_I + KX_I.T
    return torch.sqrt(KX.clamp(min=0.0))


def euclidean_distance_matrix3(matrix: torch.Tensor):
    X_norm = torch.sum(matrix ** 2, dim=1)

    # Compute the squared Euclidean distances for all pairs of points
    # X_norm is reshaped to (N, 1) to broadcast along rows, and (1, N) to broadcast along columns
    distances = X_norm.view(-1, 1) + X_norm.view(1, -1) - 2.0 * torch.matmul(matrix, matrix.T)

    # Ensure distances are non-negative due to floating-point arithmetic errors
    # gradients will fail if we don't clamp the distances
    distances = torch.clamp(distances, min=1e-12)

    return torch.sqrt(distances)
