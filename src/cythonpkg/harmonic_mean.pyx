"""
Calculate harmonic_mean:
$$
H = \frac{N_{measurements}}{\sum^{N}_{n=1}{\frac{1}{m_n}}
$$
"""


def harmonic_mean(nums):
    """H = \frac{N_{measurements}}{\sum^{N}_{n=1}{\frac{1}{m_n}}
    """
    return len(nums) / sum(1 / num for num in nums)
