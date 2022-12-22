from src.utils import compose


def test_composition():
    # Given
    fun1 = lambda x: x + 1
    fun2 = lambda x: x * 2
    given = 3
    expected = fun2(fun1(given))
    # When
    new_func = compose([fun1, fun2])
    result = new_func(given)
    # Then
    assert result == expected


def test_composition_3():
    # Given
    fun1 = lambda x: x + 1
    fun2 = lambda x: x * 2
    fun3 = lambda x: x**2
    given = 3
    expected = fun3(
                    fun2(
                        fun1(given)
                    )
                )
    # When
    new_func = compose([fun1, fun2, fun3])
    result = new_func(given)
    # Then
    assert result == expected