import torch
import math
from sauce.utils.distance_matrix import (euclidean_distance_matrix,
                                        euclidean_distance_matrix2,
                                        euclidean_distance_matrix3)


""""
optional bandwith selection methods (see stats R package)
"""
def quartile_value(tensor, q):
    """Calculate the quartile value for a given tensor and quantile."""
    return torch.quantile(tensor, q)


def vector_sd(tensor):
    """Calculate the standard deviation for a given tensor."""
    return torch.std(tensor, unbiased=False)


def bandwidth_selection_vector_stats_bwnrd0(conditional_vec: torch.Tensor):
    """Calculate the bandwidth using the rule of thumb."""
    iqr = quartile_value(conditional_vec, 0.75) - quartile_value(conditional_vec, 0.25)
    sd = vector_sd(conditional_vec)
    lo = torch.min(sd, iqr / 1.34)
    return 0.9 * lo * pow(conditional_vec.numel(), -0.2)

def bandwidth_selection(conditional_input: torch.Tensor) -> torch.Tensor:
    """Bandwidth selection for vectors or matrices."""
    if conditional_input.dim() == 1 or conditional_input.size(1) == 1:
        # For a vector
        bandwidth = bandwidth_selection_vector_stats_bwnrd0(conditional_input)
        return bandwidth if bandwidth > 0.0 else torch.tensor(0.5)
    else:
        # For a matrix, return a vector of bandwidths
        bandwidth = torch.tensor(list(map(bandwidth_selection_vector_stats_bwnrd0, conditional_input.t())))

    return bandwidth
##############################################################


def centered_weight_distance_matrix(distance_matrix: torch.Tensor, weight: torch.Tensor):
    assert distance_matrix.dim() == 2, "Distance matrix must be a 2D tensor."
    assert weight.dim() == 1, "Weight must be a 1D tensor."
    weight_sum = torch.sum(weight)
    
    marginal_weight_distance = torch.matmul(distance_matrix, weight) / (weight_sum + 1e-12)

    weight_distance_sum = torch.dot(marginal_weight_distance, weight) / (weight_sum + 1e-12)

    weight_distance_centered = distance_matrix - marginal_weight_distance.unsqueeze(1) - marginal_weight_distance + weight_distance_sum
    return weight_distance_centered


def centered_weight_distance_matrix_batch(distance_matrix: torch.Tensor, weight: torch.Tensor):
    # Ensure the input tensors have the expected dimensions
    assert distance_matrix.dim() == 2, "Distance matrix must be a 2D tensor."
    assert weight.dim() == 2, "Weight must be a 2D tensor where each row is a separate set of weights."

    weight_sum = torch.sum(weight, dim=1, keepdim=True)

    # Adjust calculations to work with batches of weights
    marginal_weight_distance = torch.matmul(distance_matrix, weight.t()) / (weight_sum.t() + 1e-12)

    weight_distance_sum = torch.sum(marginal_weight_distance * weight.t(), dim=0) / (weight_sum.squeeze() + 1e-12)

    weight_distance_centered = distance_matrix.unsqueeze(2) - marginal_weight_distance.unsqueeze(1) \
                                  - marginal_weight_distance.unsqueeze(0) + weight_distance_sum

    return weight_distance_centered


def compute_conditional_distance_correlation_from_distances(distance_x: torch.Tensor,
                                           distance_y: torch.Tensor,
                                           kernel_density_estimation: torch.Tensor):
    assert distance_x.dim() == 2, "Distance matrix X must be a 2D tensor."
    assert distance_y.dim() == 2, "Distance matrix Y must be a 2D tensor."
    assert kernel_density_estimation.dim() == 2, "Kernel density estimation must be a 2D tensor."
    num = distance_x.size(0)
    condition_distance_covariance_xy = torch.zeros(num, device=distance_x.device)
    condition_distance_covariance_xx = torch.zeros(num, device=distance_x.device)
    condition_distance_covariance_yy = torch.zeros(num, device=distance_x.device)

    for i in range(num):
        centered_weight_dist_x = centered_weight_distance_matrix(distance_x, kernel_density_estimation[i])
        centered_weight_dist_y = centered_weight_distance_matrix(distance_y, kernel_density_estimation[i])

        kde_i = kernel_density_estimation[i].unsqueeze(-1)
        kde_i_product = kde_i @ kde_i.transpose(0, 1)

        condition_distance_covariance_xy[i] = torch.sum(centered_weight_dist_x * centered_weight_dist_y * kde_i_product)
        condition_distance_covariance_xx[i] = torch.sum(centered_weight_dist_x * centered_weight_dist_x * kde_i_product)
        condition_distance_covariance_yy[i] = torch.sum(centered_weight_dist_y * centered_weight_dist_y * kde_i_product)

    dcor_denominator = condition_distance_covariance_xx * condition_distance_covariance_yy
    valid_denominator = dcor_denominator > 0.0
    condition_distance_covariance_xy /= (torch.sqrt(dcor_denominator.clamp(1e-12)))

    return condition_distance_covariance_xy


def compute_conditional_distance_correlation_batch_from_distances(distance_x: torch.Tensor,
                                                 distance_y: torch.Tensor,
                                                 kernel_density_estimation: torch.Tensor):
    assert distance_x.dim() == 2 and distance_y.dim() == 2, "Distance matrices must be 2D."
    assert kernel_density_estimation.dim() == 2, "Kernel density estimation must be a 2D tensor."
    
    # Calculate ANOVA tables for all kernel density estimates in parallel
    centered_weight_dist_x_batch = centered_weight_distance_matrix_batch(distance_x, kernel_density_estimation)
    centered_weight_dist_y_batch = centered_weight_distance_matrix_batch(distance_y, kernel_density_estimation)

    condition_distance_covariance_xy = torch.sum(centered_weight_dist_x_batch *
                                                 centered_weight_dist_y_batch *
                                                 kernel_density_estimation.unsqueeze(0) *
                                                 kernel_density_estimation.unsqueeze(1), dim=(0, 1))
    condition_distance_covariance_xx = torch.sum(centered_weight_dist_x_batch *
                                                 centered_weight_dist_x_batch *
                                                 kernel_density_estimation.unsqueeze(0) *
                                                 kernel_density_estimation.unsqueeze(1), dim=(0, 1))
    condition_distance_covariance_yy = torch.sum(centered_weight_dist_y_batch *
                                                 centered_weight_dist_y_batch *
                                                 kernel_density_estimation.unsqueeze(0) *
                                                 kernel_density_estimation.unsqueeze(1), dim=(0, 1))

    dcor_denominator = condition_distance_covariance_xx * condition_distance_covariance_yy
    condition_distance_covariance_xy /= torch.sqrt(dcor_denominator.clamp(min=1e-12))

    return condition_distance_covariance_xy


def compute_gaussian_kernel_estimate(condition_variable: torch.Tensor, bandwidth: torch.Tensor):
    assert condition_variable.dim() == 2, "Condition variable must be a 2D tensor."
    assert bandwidth.dim() == 0 or bandwidth.dim() == 1, "Bandwidth must be a scalar or a 1D tensor."
    num, d = condition_variable.shape
    det = torch.prod(bandwidth) if bandwidth.dim() == 1 else torch.pow(bandwidth, d)

    density = 1.0 / (math.pow(2 * math.pi, d / 2.0) * det)

    weight = 1.0 / bandwidth**2
    diff = condition_variable.unsqueeze(1) - condition_variable.unsqueeze(0)

    quadric_value = torch.sum((diff ** 2) * weight, dim=2)

    kernel_density_estimate = torch.exp(-0.5 * quadric_value) * density

    return kernel_density_estimate


def gaussian_kde(X, sigma: torch.Tensor | float):
    assert sigma is None or isinstance(sigma, (torch.Tensor, float)), "Sigma must be a scalar or a 1D tensor."
    sigma = torch.tensor(sigma) if isinstance(sigma, float) else sigma
    _, d = X.shape
    GX = torch.matmul(X, X.T)
    KX = (torch.diag(GX) - GX)
    KX = KX + KX.T
    det = torch.prod(sigma) if sigma.dim() == 1 else torch.pow(sigma, d)
    density = 1.0 / (math.pow(2 * math.pi, d / 2.0) * det)
    return torch.exp(-0.5 * KX / (sigma**2)) * density


def gaussian_kde2(X: torch.Tensor, sigma: torch.Tensor):
    assert sigma is None or isinstance(sigma, (torch.Tensor, float)), "Sigma must be a scalar or a 1D tensor."
    _, d = X.shape
    X_norm = torch.sum(X ** 2, dim=1)

    # Compute the squared Euclidean distances for all pairs of points
    # X_norm is reshaped to (N, 1) to broadcast along rows, and (1, N) to broadcast along columns
    distances = X_norm.view(-1, 1) + X_norm.view(1, -1) - 2.0 * torch.matmul(X, X.T)

    # Ensure distances are non-negative due to floating-point arithmetic errors
    KX = torch.clamp(distances, min=0.0)
    det = torch.prod(sigma) if sigma.dim() == 1 else torch.pow(sigma, d)
    norm_factor = 1.0 / (math.pow(2 * math.pi, d / 2.0) * det)
    return torch.exp(-0.5 * KX / (sigma*sigma)) * norm_factor


class ConditionalDistanceCorrelation(torch.nn.Module):
    def __init__(self, bandwidth: torch.Tensor | float, *args, **kwargs) -> None:
        super().__init__()
        self.bandwidth = torch.tensor(bandwidth) if isinstance(bandwidth, float) else bandwidth

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) < 3 and len(y.shape) < 3 and len(z.shape) < 3, "X, Y, and Z must be 2D or 1D tensors."
        x = x if len(x.shape) == 2 else x.unsqueeze(1)
        y = y if len(y.shape) == 2 else y.unsqueeze(1)
        z = z if len(z.shape) == 2 else z.unsqueeze(1)
        distance_x = euclidean_distance_matrix3(x)
        distance_y = euclidean_distance_matrix3(y)
        kde_z = gaussian_kde2(z, self.bandwidth)
        return torch.mean(compute_conditional_distance_correlation_batch_from_distances(distance_x, distance_y, kde_z))


if __name__ == "__main__":
    import time

    # # Test the implementation
    # x = torch.randn(128, 256, device="cuda")
    # y = torch.randn(128, 256, device="cuda")
    # z = torch.randn(128, 256, device="cuda")

    # bandwidth = 0.4

    # distance_x = euclidean_distance_matrix(x)
    # distance_y = euclidean_distance_matrix(y)

    # kde_z = gaussian_kde(z, bandwidth)

    # result = compute_conditional_distance_correlation_from_distances(distance_x, distance_y, kde_z)

    # result2 = compute_conditional_distance_correlation_batch_from_distances(distance_x, distance_y, kde_z)

    # # result3 = compute_condition_distance_correlation_batch_old(distance_x, distance_y, kde_z)
    
    # print(torch.allclose(result, result2))
    # # print(torch.allclose(result, result3))
    # # print(torch.allclose(result2, result3))

    # start = time.time()
    # for _ in range(100):
    #     result = compute_conditional_distance_correlation_from_distances(distance_x, distance_y, kde_z)
    # print("Elapsed time for single: ", time.time() - start)

    # start = time.time()
    # for _ in range(100):
    #     result2 = compute_conditional_distance_correlation_batch_from_distances(distance_x, distance_y, kde_z)
    # print("Elapsed time for batch2: ", time.time() - start)

    # start = time.time()
    # for _ in range(100):
    #     result3 = compute_condition_distance_correlation_batch_old(distance_x, distance_y, kde_z)
    # print("Elapsed time for batch: ", time.time() - start)

    # X = torch.randn(500, 10, device="cuda", requires_grad=True)
    # Y = torch.randn(500, 10, device="cuda", requires_grad=True)
    # Z = torch.randn(500, 10, device="cuda", requires_grad=True)

    # cdc = ConditionalDistanceCorrelation()

    # cdcor = cdc(X, Y, Z)
    # cdcor.backward()

    # print(X.grad)
    # print(Y.grad)
    # print(Z.grad)

    x = torch.randn(128, 1)
    print(euclidean_distance_matrix3(x))
