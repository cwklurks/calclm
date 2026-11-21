import numpy as np
import pytest

from model.pack import lut_matvec, pack_ternary, unpack_ternary


def test_single_group_byte_value():
    # first weight is the least significant base-3 digit
    w = np.array([[1, 0, -1, 0, 1]], dtype=np.int8)
    assert pack_ternary(w).tolist() == [[2 + 1 * 3 + 0 * 9 + 1 * 27 + 2 * 81]]


def test_all_zero_and_all_one_bytes():
    assert pack_ternary(np.zeros((1, 5), np.int8)).tolist() == [[121]]
    assert pack_ternary(np.ones((1, 5), np.int8)).tolist() == [[242]]
    assert pack_ternary(-np.ones((1, 5), np.int8)).tolist() == [[0]]


def test_pads_inputs_to_multiple_of_five_with_zero_weights():
    w = np.array([[1, -1, 1, -1, 1, 1, -1]], dtype=np.int8)
    packed = pack_ternary(w)
    assert packed.shape == (1, 2)
    assert unpack_ternary(packed, 7).tolist() == w.tolist()
    assert unpack_ternary(packed, 10)[0, 7:].tolist() == [0, 0, 0]


@pytest.mark.parametrize("rows,cols", [(1, 1), (3, 5), (192, 192), (512, 192), (192, 512)])
def test_roundtrip(rows, cols):
    rng = np.random.default_rng(rows * 1000 + cols)
    w = rng.integers(-1, 2, size=(rows, cols), dtype=np.int8)
    assert np.array_equal(unpack_ternary(pack_ternary(w), cols), w)


def test_rejects_non_ternary():
    with pytest.raises(ValueError):
        pack_ternary(np.array([[2]], dtype=np.int8))


@pytest.mark.parametrize("rows,cols", [(7, 13), (192, 192), (512, 192), (192, 512)])
def test_lut_matvec_matches_dense(rows, cols):
    rng = np.random.default_rng(cols)
    w = rng.integers(-1, 2, size=(rows, cols), dtype=np.int8)
    x = rng.integers(-127, 128, size=cols, dtype=np.int8)
    want = w.astype(np.int32) @ x.astype(np.int32)
    assert lut_matvec(pack_ternary(w), x).tolist() == want.tolist()


def test_lut_matvec_extremes_fit_24_bits():
    w = np.ones((1, 512), np.int8)
    x = np.full(512, -127, np.int8)
    y = lut_matvec(pack_ternary(w), x)
    assert y.tolist() == [-127 * 512]
    assert -(1 << 23) <= int(y[0]) < (1 << 23)
