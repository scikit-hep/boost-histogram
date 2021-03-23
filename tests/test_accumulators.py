import pytest
from pytest import approx

import boost_histogram as bh

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")


def test_weighted_sum():
    a = bh.accumulators.WeightedSum(1.5, 2.5)

    assert repr(a) == "WeightedSum(value=1.5, variance=2.5)"

    assert a == bh.accumulators.WeightedSum(1.5, 2.5)

    assert a.value == 1.5
    assert a.variance == 2.5

    a += 1.5

    assert a.value == 3.0
    assert a.variance == 4.75

    vals = [1, 2, 3]
    vari = [4, 5, 6]

    a = bh.accumulators.WeightedSum()
    for val, var in zip(vals, vari):
        a += bh.accumulators.WeightedSum(val, variance=var)

    assert a.value == 6
    assert a.variance == 15

    a2 = bh.accumulators.WeightedSum().fill(vals, variance=vari)
    assert a == a2

    assert a == bh.accumulators.WeightedSum(6, 15)


def test_sum():
    vals = [1, 2, 3]
    a = bh.accumulators.Sum()
    for val in vals:
        a += val

    assert a.value == 6

    a2 = bh.accumulators.Sum().fill(vals)
    assert a == a2

    assert a == bh.accumulators.Sum(6)

    assert repr(a) == "Sum(6 + 0)"


def test_weighted_mean():
    vals = [4, 1]
    weights = [1, 2]
    a = bh.accumulators.WeightedMean()
    for v, w in zip(vals, weights):
        a(v, weight=w)

    assert a.sum_of_weights == 3.0
    assert a.variance == 4.5
    assert a.value == 2.0

    a2 = bh.accumulators.WeightedMean().fill(vals, weight=weights)
    assert a == a2

    assert a == bh.accumulators.WeightedMean(3, 5, 2, 4.5)
    assert (
        repr(a)
        == "WeightedMean(sum_of_weights=3, sum_of_weights_squared=5, value=2, variance=4.5)"
    )


def test_mean():
    vals = [1, 2, 3]
    a = bh.accumulators.Mean()
    for val in vals:
        a(val)

    assert a.count == 3
    assert a.value == 2
    assert a.variance == 1

    a2 = bh.accumulators.Mean().fill([1, 2, 3])
    assert a == a2

    assert a == bh.accumulators.Mean(3, 2, 1)

    assert repr(a) == "Mean(count=3, value=2, variance=1)"


float_st = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e4, max_value=1e4
)
simple_list_st = st.lists(float_st, min_size=1, max_size=10)


@hypothesis.given(
    simple_list_st,
    simple_list_st,
)
def test_sum_mean(list1, list2):
    a = bh.accumulators.Mean()
    a.fill(list1)

    b = bh.accumulators.Mean()
    b.fill(list2)

    c = bh.accumulators.Mean()
    c.fill(list1 + list2)

    ab = a + b
    assert ab.value == approx(c.value)
    assert ab.variance == approx(c.variance, nan_ok=True, abs=1e-7, rel=1e-3)
    assert ab.count == approx(c.count)

    a += b
    assert a.value == approx(c.value)
    assert a.variance == approx(c.variance, nan_ok=True, abs=1e-7, rel=1e-3)
    assert a.count == approx(c.count)


dual_lists_st = st.integers(min_value=1, max_value=10).flatmap(
    lambda n: st.tuples(
        st.lists(float_st, min_size=n, max_size=n),
        st.lists(
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=1e-2, max_value=1e2
            ),
            min_size=n,
            max_size=n,
        ),
    )
)


@hypothesis.given(dual_lists_st, dual_lists_st)
def test_sum_weighed_mean(pair1, pair2):
    a = bh.accumulators.WeightedMean()
    a.fill(pair1[0], weight=pair1[1])

    b = bh.accumulators.WeightedMean()
    b.fill(pair2[0], weight=pair2[1])

    c = bh.accumulators.WeightedMean()
    c.fill(pair1[0] + pair2[0], weight=pair1[1] + pair2[1])

    ab = a + b
    assert ab.value == approx(c.value)
    assert ab.variance == approx(c.variance, nan_ok=True, abs=1e-3, rel=1e-2)
    assert ab.sum_of_weights == approx(c.sum_of_weights)
    assert ab.sum_of_weights_squared == approx(c.sum_of_weights_squared)

    a += b
    assert a.value == approx(c.value)
    assert a.variance == approx(c.variance, nan_ok=True, abs=1e-3, rel=1e-2)
    assert a.sum_of_weights == approx(c.sum_of_weights)
    assert a.sum_of_weights_squared == approx(c.sum_of_weights_squared)
