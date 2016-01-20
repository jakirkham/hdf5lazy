from hdf5lazy import H5PyDataset, NetCDFDataset
import h5py
import netCDF4
import numpy as np
import os
import pytest
import pickle


h5py_path = 'tmp.hdf5'
netcdf_path = 'sresa1b_ncar_ccsm3-example.nc'

data = {'x': np.eye(3),
        'y': np.arange(5),
        '/data/ones': np.ones(5)}


@pytest.yield_fixture
def h5():
    if os.path.exists(h5py_path):
        os.remove(h5py_path)
    with h5py.File(h5py_path) as f:
        for k, v in data.items():
            f[k] = v

        f.flush()

        yield f


def test_NetCDFDataset():
    a = NetCDFDataset(netcdf_path, 'tas')
    b = netCDF4.Dataset(netcdf_path).variables['tas']

    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert isinstance(a[:, 0, :], np.ndarray)
    assert (a[0, :5, 5:10] == b[0, :5, 5:10]).all()

    c = NetCDFDataset(netcdf_path, 'tas')
    assert a.dataset is c.dataset


def test_H5PyDataset(h5):
    a = H5PyDataset(h5py_path, 'x')
    b = h5['x']

    assert (a[:, :] == b[:, :]).all()
    assert (a[1, :] == b[1, :]).all()
    assert (a[1] == b[1]).all()

    a[0, 1] = 5
    assert b[0, 1] == 5

    try:
        a[1, 1, 1, 1]
    except Exception as e:
        ee = e

    try:
        b[1, 1, 1, 1]
    except Exception as f:
        ff = f

    assert str(ee) == str(ff)


def test_pickle(h5):
    a = H5PyDataset(h5py_path, 'x')
    b = pickle.loads(pickle.dumps(a))

    assert a.dataset is b.dataset
    assert a.datapath == b.datapath


def test_object_equivalence(h5):
    a = H5PyDataset(h5py_path, 'x')
    b = H5PyDataset(h5py_path, 'x')
    assert a.path == h5py_path
    assert a.dataset is b.dataset
