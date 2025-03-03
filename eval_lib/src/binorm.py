################################################################
### Bivariate Normal CDF
# using jax implementation: https://github.com/jax-ml/jax/issues/10562
# from the paper "A simple approximation for the bivariate normal integral", Tsay & Ke (2021)
# https://www.tandfonline.com/doi/full/10.1080/03610918.2021.1884718

import torch as t
from torch import erf

cdf1d = t.distributions.Normal(0, 1).cdf

c1 = -1.0950081470333
c2 = -0.75651138383854
sqrt2 = 1.4142135623730951

def case1(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * t.sqrt(1 - a2 * c2))) * t.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * t.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * t.sqrt(1 - a2 * c2))) * t.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux5 = 2 * t.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * t.sqrt(1 - a2 * c2)))

    return line11 + (line12 * line21) - (line22 * (line31 + line32))


def case2(p, q):
    return cdf1d(p) * cdf1d(q)


def case3(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * t.sqrt(1 - a2 * c2))) * t.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux4 = 2 * t.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12


def case4(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * t.sqrt(1 - a2 * c2))) * t.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux4 = 2 * t.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)


def case5(p, q, rho, a, b):
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * t.sqrt(1 - a2 * c2))) * t.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * t.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * t.sqrt(1 - a2 * c2))) * t.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux5 = 2 * t.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * t.sqrt(1 - a2 * c2)))

    return line11 - (line12 * line21) + line22 * (line31 + line32)



def binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    '''
    Compute the bivariate normal CDF.

    Parameters:
    ----------
    x1, x2: int, float, t.Tensor
        The values at which to compute the CDF.
    mu1, mu2: int, float, t.Tensor
        The margianal means of the normal distribution.
    sigma1, sigma2: int, float, t.Tensor
        The marginal standard deviations of the normal distribution.
    rho: int, float, t.Tensor
        The correlation coefficient of the normal distribution.
    '''
    # Make sure the inputs are either int, float or t.Tensor.
    # If they're float, convert them to singleton float t.Tensors.
    # If they're t.Tensor, make sure they're of the same dtype and shape.
    shape = None
    dtype = None
    device = None
    for input_ in [x1, x2, mu1, mu2, sigma1, sigma2, rho]:
        if isinstance(input_, t.Tensor):
            if shape is None:
                shape = input_.shape
            else:
                assert shape == input_.shape, f"All t.Tensors must have the same shape. Got {input_.shape} and {shape}."
            if dtype is None:
                dtype = input_.dtype
            else:
                assert dtype == input_.dtype, f"All t.Tensors must have the same dtype. Got {input_.dtype} and {dtype}."
            if device is None:
                device = input_.device
            else:
                assert device == input_.device, f"All t.Tensors must be on the same device. Got {input_.device} and {device}."
        else:
            assert isinstance(input_, (int, float)), f"All inputs must be either float or t.Tensor. Got {type(input_)}."
    
    if shape is None:
        shape = (1,)

    # Convert the inputs to t.Tensor if they're int or float.
    x1     = t.tensor([float(x1)],     dtype=dtype).to(device).expand(*shape) if isinstance(x1,     (int, float)) else x1
    x2     = t.tensor([float(x2)],     dtype=dtype).to(device).expand(*shape) if isinstance(x2,     (int, float)) else x2
    mu1    = t.tensor([float(mu1)],    dtype=dtype).to(device).expand(*shape) if isinstance(mu1,    (int, float)) else mu1
    mu2    = t.tensor([float(mu2)],    dtype=dtype).to(device).expand(*shape) if isinstance(mu2,    (int, float)) else mu2
    sigma1 = t.tensor([float(sigma1)], dtype=dtype).to(device).expand(*shape) if isinstance(sigma1, (int, float)) else sigma1
    sigma2 = t.tensor([float(sigma2)], dtype=dtype).to(device).expand(*shape) if isinstance(sigma2, (int, float)) else sigma2
    rho    = t.tensor([float(rho)],    dtype=dtype).to(device).expand(*shape) if isinstance(rho,    (int, float)) else rho

    p = (x1 - mu1) / sigma1
    q = (x2 - mu2) / sigma2

    a = -rho / t.sqrt(1 - rho * rho)
    b = p / t.sqrt(1 - rho * rho)

    assert a.shape == b.shape == p.shape == q.shape == rho.shape == shape, f"Shapes of a, b, p, q and rho must be the same. Got {a.shape}, {b.shape}, {p.shape}, {q.shape} and {rho.shape}. Should be {shape}."

    # find the indices where each case applies
    case1_indices = (a > 0) & (a * q + b >= 0)
    case2_indices = (a == 0)
    case3_indices = (a > 0) & (a * q + b < 0)
    case4_indices = (a < 0) & (a * q + b >= 0)
    case5_indices = (a < 0) & (a * q + b < 0)

    # compute the CDF for each case
    result = t.zeros_like(p)

    result[case1_indices] = case1(p[case1_indices], q[case1_indices], rho[case1_indices], a[case1_indices], b[case1_indices])
    result[case2_indices] = case2(p[case2_indices], q[case2_indices])
    result[case3_indices] = case3(p[case3_indices], q[case3_indices], rho[case3_indices], a[case3_indices], b[case3_indices])
    result[case4_indices] = case4(p[case4_indices], q[case4_indices], rho[case4_indices], a[case4_indices], b[case4_indices])
    result[case5_indices] = case5(p[case5_indices], q[case5_indices], rho[case5_indices], a[case5_indices], b[case5_indices])

    return result
    

if __name__ == "__main__":
    x1 = 0.0
    x2 = 0.0
    mu1 = 0.0
    mu2 = 0.0
    sigma1 = 1.0
    sigma2 = 1.0
    rho = 0.5

    out = binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho)
    print(out, out.shape)

    x1 = t.tensor([0.0, 0.0, 0.0])
    x2 = t.tensor([0.0, 0.0, 0.0])
    mu1 = t.tensor([0.0, 0.0, 0.0])
    mu2 = t.tensor([0.0, 0.0, 0.0])
    sigma1 = t.tensor([1.0, 1.0, 1.0])
    sigma2 = t.tensor([1.0, 1.0, 1.0])
    rho = t.tensor([0.5, 0.5, 0.5])

    out = binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho)
    print(out, out.shape)

    x1 = t.tensor([0.0, 0.0, 0.0])
    x2 = t.tensor([0.0, 0.0, 0.0])
    mu1 = 0.0
    mu2 = 0.0
    sigma1 = 1.0
    sigma2 = 1.0
    rho = 0.5

    out = binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho)
    print(out, out.shape)

    x1 = 0.0
    x2 = 0.0
    mu1 = t.tensor([0.0, 0.0, 0.0])
    mu2 = t.tensor([0.0, 0.0, 0.0])
    sigma1 = 1.0
    sigma2 = 1.0
    rho = t.tensor([0.5, 0.5, 0.5])

    out = binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho)
    print(out, out.shape)
    
    x1 = 0.0
    x2 = 0.0
    mu1 = t.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    mu2 = t.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sigma1 = 1.0
    sigma2 = 1.0
    rho = t.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

    out = binorm_cdf(x1, x2, mu1, mu2, sigma1, sigma2, rho)
    print(out, out.shape)
    
