import matha
import pytest

def test_sanity():
    """Basic check to ensure the library loads."""
    assert 1 + 1 == 2

def test_trig_sin():
    """Testing the @_angle_handler decorator in trigonometry.py."""
    # 90 degrees should be 1.0
    assert matha.sin(90, in_degrees=True) == pytest.approx(1.0)

def test_numtheory_prime():
    """Testing the is_prime function in numTheory.py."""
    assert matha.is_prime(7) is True
    assert matha.is_prime(10) is False

def test_complex_creation():
    """Testing the create function in complex.py."""
    z = matha.create(3, 4)
    assert z.real == 3
    assert z.imag == 4

def test_matrix_multiplication():
    """Testing the matrix_multiply function in matrix.py."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = matha.matrix_multiply(A, B)
    expected = [[19, 22], [43, 50]]
    assert result == expected

def test_vector_dot_product():
    """Testing the vector_dot function in vector.py."""
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    result = matha.vector_dot(v1, v2)
    expected = 32  # 1*4 + 2*5 + 3*6
    assert result == expected

def test_prime_factors():
    """Testing the prime_factors function in numTheory.py."""
    n = 28
    result = matha.prime_factors(n)
    expected = [2, 7]
    assert result == expected

def test_nth_prime():
    """Testing the nth_prime function in numTheory.py."""
    result = matha.nth_prime(2000)
    expected = 17389
    assert result == expected

