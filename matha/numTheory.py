# numTheory.py

# -------------------------
# @Author : AstroJr0
# @Date : 16-12-2025
# -------------------------

import math
import numpy as np # Used for math.gcd in Python standard library alternative

# --- 1. PRIMALITY & PRIME FUNCTIONS ---

def is_prime(n: int) -> bool:
    """
    Checks if a positive integer n is a prime number.
    Uses an optimized trial division check based on the fact that primes 
    > 3 must be of the form 6k ± 1. 
    
    Args:
        n: The integer to check.
        
    Returns:
        True if n is prime, False otherwise.
    """
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# --- 2. FACTORS & PRIME FACTORIZATION ---

def prime_factors(n: int) -> dict:
    """
    Calculates the prime factorization of a positive integer n.
    
    Args:
        n: The integer to factorize.
        
    Returns:
        A dictionary mapping prime factors to their exponents (e.g., 12 -> {2: 2, 3: 1}).
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    factors = {}
    
    # Factor out all 2s
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    # Factor out odd primes starting from 3
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors[f] = factors.get(f, 0) + 1
            n //= f
        f += 2

    # If n remains > 1, it must be the last (largest) prime factor
    if n > 1:
        factors[n] = 1

    return factors


# --- 3. EUCLIDEAN ALGORITHMS & LCM ---

def gcd(a: int, b: int) -> int:
    """
    Calculates the Greatest Common Divisor (GCD) of two integers using 
    the Euclidean Algorithm. 
    
    Args:
        a: The first integer.
        b: The second integer.
        
    Returns:
        The greatest common divisor.
    """
    # Uses the iterative (modulo) method
    while b:
        a, b = b, a % b
    return abs(a)


def extended_gcd(a: int, b: int) -> tuple:
    """
    Implements the Extended Euclidean Algorithm to find integers x and y 
    such that a*x + b*y = gcd(a, b). 
    
    Args:
        a: The first integer.
        b: The second integer.
        
    Returns:
        A tuple (g, x, y) where g = gcd(a, b).
    """
    if a == 0:
        return (b, 0, 1)
    
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return (g, x, y)


def lcm(a: int, b: int) -> int:
    """
    Calculates the Least Common Multiple (LCM) of two integers.
    Uses the relationship: lcm(a, b) = |a * b| / gcd(a, b).
    
    Args:
        a: The first integer.
        b: The second integer.
        
    Returns:
        The least common multiple.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


# --- 4. MODULAR ARITHMETIC ---

def mod_pow(base: int, exp: int, mod: int) -> int:
    """
    Calculates (base^exp) % mod efficiently using the method of 
    exponentiation by squaring (binary exponentiation). 
    
    Args:
        base: The base number.
        exp: The exponent (must be non-negative).
        mod: The modulus.
        
    Returns:
        The result of the modular exponentiation.
    """
    if exp < 0:
        raise ValueError("Exponent must be non-negative.")
        
    result = 1
    base %= mod
    
    while exp > 0:
        # If exp is odd, multiply the result by base
        if exp & 1:
            result = (result * base) % mod
            
        # exp = exp / 2 (right shift) and base = base^2
        exp >>= 1
        base = (base * base) % mod
        
    return result


def mod_inverse(a: int, mod: int) -> int:
    """
    Calculates the modular multiplicative inverse of 'a' modulo 'mod'.
    i.e., finds x such that (a * x) % mod = 1.
    
    Args:
        a: The integer to invert.
        mod: The modulus (must be positive).
        
    Returns:
        The modular inverse x.
        
    Raises:
        ValueError: If the inverse does not exist (a and mod are not coprime).
    """
    g, x, y = extended_gcd(a, mod)
    
    if g != 1:
        # The inverse exists only if gcd(a, mod) = 1
        raise ValueError(f"Modular inverse of {a} mod {mod} does not exist (GCD is {g}).")
    
    # x might be negative, so adjust to be positive in [0, mod-1] range
    return x % mod


# --- 5. NUMBER SEQUENCES & PROPERTIES ---

def _fib_fast_doubling(n: int) -> tuple:
    """
    Helper function for fast O(log n) calculation of Fibonacci numbers 
    using the matrix/doubling identity: F(2n) = F(n) * (2*F(n+1) - F(n))
    """
    if n == 0:
        return (0, 1) # Returns (F(0), F(1))
        
    a, b = _fib_fast_doubling(n // 2) # a=F(n/2), b=F(n/2 + 1)
    
    # Calculate F(2n) and F(2n + 1)
    c = a * (2 * b - a)
    d = a * a + b * b
    
    if n % 2 == 0:
        return (c, d) # Returns (F(2k), F(2k + 1))
    else:
        return (d, c + d) # Returns (F(2k + 1), F(2k + 2))


def fibonacci(n: int) -> int:
    """
    Calculates the n-th Fibonacci number F(n) using the fast doubling method (O(log n)).
    
    Args:
        n: The index of the Fibonacci number (0-indexed: F(0)=0, F(1)=1, F(2)=1...).
        
    Returns:
        The n-th Fibonacci number.
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is typically defined for non-negative indices.")
    return _fib_fast_doubling(n)[0]


def totient(n: int) -> int:
    """
    Calculates Euler's Totient function (φ(n)), which counts the number of positive 
    integers up to n that are relatively prime to n (i.e., gcd(k, n) = 1).
    
    Uses the formula: φ(n) = n * product_{p|n} (1 - 1/p), where p are the distinct 
    prime factors of n. 
    
    Args:
        n: The integer (must be positive).
        
    Returns:
        The value of φ(n).
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    if n == 1:
        return 1
        
    result = n
    pf = prime_factors(n)
    
    for p in pf:
        # result = result * (1 - 1/p)
        result -= result // p 
        
    return result


def is_perfect(n: int) -> bool:
    """
    Checks if a positive integer n is a perfect number.
    A perfect number is a positive integer that is equal to the sum of its 
    proper positive divisors (divisors excluding the number itself).
    
    Args:
        n: The integer to check.
        
    Returns:
        True if n is perfect (e.g., 6, 28, 496), False otherwise.
    """
    if n < 2: return False
    
    s = 1 # Start sum with 1 (always a proper divisor)
    i = 2
    
    while i * i <= n:
        if n % i == 0:
            s += i
            # Add the corresponding paired divisor if it's not the square root
            if i != n // i:
                s += n // i
        i += 1
        
    return s == n

def nth_prime(n: int) -> int:
    """
    This is the most basic function to return the n-th prime number (1-indexed).
    
    Args:
        n: The index of the prime number to retrieve (1-indexed).
    
    Returns:
        The n-th prime number.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    
    count = 0
    candidate = 1
    
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
            
    return candidate
