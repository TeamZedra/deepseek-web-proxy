"""Fibonacci sequence implementations in Python."""


def fib_iterative(n: int) -> list[int]:
    """Return the first n Fibonacci numbers using iteration.

    Args:
        n: How many Fibonacci numbers to generate.

    Returns:
        A list containing the first n Fibonacci numbers.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []

    sequence = [0]
    a, b = 0, 1
    for _ in range(1, n):
        a, b = b, a + b
        sequence.append(a)
    return sequence


def fib_generator(n: int):
    """Yield the first n Fibonacci numbers lazily."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


def fib_memo(n: int, _cache: dict[int, int] | None = None) -> int:
    """Return the nth Fibonacci number (0-indexed) using memoization."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if _cache is None:
        _cache = {}
    if n in _cache:
        return _cache[n]
    if n < 2:
        return n
    result = fib_memo(n - 1, _cache) + fib_memo(n - 2, _cache)
    _cache[n] = result
    return result


if __name__ == "__main__":
    count = 10
    print(f"First {count} Fibonacci numbers (iterative):")
    print(fib_iterative(count))

    print(f"\nFirst {count} Fibonacci numbers (generator):")
    print(list(fib_generator(count)))

    print(f"\nFibonacci number at index {count} (memoized):")
    print(fib_memo(count))