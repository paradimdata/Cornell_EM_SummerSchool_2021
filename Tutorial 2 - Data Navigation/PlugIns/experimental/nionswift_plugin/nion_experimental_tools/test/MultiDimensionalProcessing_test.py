import gettext
import unittest

import numpy
import scipy.ndimage

# local libraries
from nion.swift import Facade
from nion.data import DataAndMetadata
from nion.swift.test import TestContext
from nion.ui import TestUI
from nion.swift import Application
from nion.swift.model import DocumentModel

from nionswift_plugin.nion_experimental_tools import MultiDimensionalProcessing

_ = gettext.gettext


Facade.initialize()


def create_memory_profile_context() -> TestContext.MemoryProfileContext:
    return TestContext.MemoryProfileContext()


class TestMultiDimensionalProcessing(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=True)
        self.app.workspace_dir = str()

    def tearDown(self):
        pass

    def test_function_apply_multi_dimensional_shifts_4d(self):
        with self.subTest("Test for a sequence of SIs, shift collection dimensions along sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, "collection")

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1], 0.0], order=1)

            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 1D collections of 2D data, shift data dimensions along sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 1, 2))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, "data")

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [0.0, shifts[i, 0], shifts[i, 1]], order=1)

            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of SIs, shift data dimensions along collection and sequence axis"):
            shape = (5, 2, 3, 4)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 1))

            shifts = numpy.linspace(0, 3, num=numpy.prod(shape[:-1])).reshape(shape[:-1])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, "data")

            shifted = numpy.empty_like(data)

            for k in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        shifted[k, i, j] = scipy.ndimage.shift(data[k, i, j], [shifts[k, i, j]], order=1)

            self.assertTrue(numpy.allclose(result.data, shifted))

    def test_function_apply_multi_dimensional_shifts_5d(self):
        with self.subTest("Test for a sequence of 4D images, shift collection dimensions along sequence axis"):
            shape = (5, 2, 3, 4, 6)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, "collection")

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1], 0.0, 0.0], order=1)

            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 4D images, shift data dimensions along sequence axis"):
            shape = (5, 2, 3, 4, 6)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))

            shifts = numpy.array([(0., 1.), (0., 2.), (0., 3.), (0., 4.), (0., 5.)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, "data")

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [0.0, 0.0, shifts[i, 0], shifts[i, 1]], order=1)

            self.assertTrue(numpy.allclose(result.data, shifted))

        with self.subTest("Test for a sequence of 4D images, shift sequence dimension along collection axis"):
            shape = (5, 2, 3, 4, 6)
            data = numpy.arange(numpy.prod(shape)).reshape(shape)
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=DataAndMetadata.DataDescriptor(True, 2, 2))

            shifts = numpy.array([(1., 1.5, 2.),
                                  (2.5, 3., 3.5)])

            result = MultiDimensionalProcessing.function_apply_multi_dimensional_shifts(xdata, shifts, "sequence")

            shifted = numpy.empty_like(data)

            for k in range(shape[1]):
                for i in range(shape[2]):
                    shifted[:, k, i] = scipy.ndimage.shift(data[:, k, i], [shifts[k, i], 0., 0.], order=1)

            self.assertTrue(numpy.allclose(result.data, shifted))

    def test_function_measure_multi_dimensional_shifts_3d(self):
        with self.subTest("Test for a sequence of 2D data, measure shift of data dimensions along sequence axis"):
            shape = (5, 100, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[1:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(data[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.array([(0., 2.), (0., 5.), (0., 10.), (0., 2.5), (0., 3.)])

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          "data",
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[reference_index]), atol=0.5))

        with self.subTest("Test for a 2D collection of 1D data, measure shift of data dimensions along collection axis"):
            shape = (5, 5, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[2:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(numpy.repeat(data[numpy.newaxis, ...], shape[1], axis=0)[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.random.rand(*shape[:2]) * 10.0

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    shifted[i, j] = scipy.ndimage.shift(data[i, j], [shifts[i, j]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 1))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          "data",
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[numpy.unravel_index(reference_index, shifts.shape)]), atol=0.5))

        with self.subTest("Test for a sequence of 2D data, measure shift of data dimensions along sequence axis relative to previous slice"):
            shape = (5, 100, 100)
            data = numpy.random.rand(*shape[1:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(data[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.array([(0., 2.), (0., 5.), (0., 10.), (0., 2.5), (0., 3.)])

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                shifted[i] = scipy.ndimage.shift(data[i], [shifts[i, 0], shifts[i, 1]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(True, 0, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          "data",
                                                                                          reference_index=None)
            expected_res = -1.0 * (shifts[1:] - shifts[:-1])
            expected_res = numpy.append(numpy.zeros((1, 2)), expected_res, axis=0)
            expected_res = numpy.cumsum(expected_res, axis=0)

            self.assertTrue(numpy.allclose(result.data, expected_res, atol=0.5))

    def test_function_measure_multi_dimensional_shifts_4d(self):
        with self.subTest("Test for a 2D collection of 2D data, measure shift of data dimensions along collection axis"):
            shape = (5, 5, 100, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[2:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(numpy.repeat(data[numpy.newaxis, ...], shape[1], axis=0)[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.random.rand(*shape[:2], 2) * 10.0

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    shifted[i, j] = scipy.ndimage.shift(data[i, j], [shifts[i, j, 0], shifts[i, j, 1]], order=1, cval=numpy.mean(data))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          "data",
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[numpy.unravel_index(reference_index, shifts.shape[:-1])]), atol=0.5))

        with self.subTest("Test for a 2D collection of 2D data, measure shift of collection dimensions along data axis"):
            shape = (5, 5, 100, 100)
            reference_index = 0
            data = numpy.random.rand(*shape[2:])
            data = scipy.ndimage.gaussian_filter(data, 3.0)
            data = numpy.repeat(numpy.repeat(data[numpy.newaxis, ...], shape[1], axis=0)[numpy.newaxis, ...], shape[0], axis=0)

            shifts = numpy.random.rand(*shape[:2], 2) * 10.0

            shifted = numpy.empty_like(data)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    shifted[i, j] = scipy.ndimage.shift(data[i, j], [shifts[i, j, 0], shifts[i, j, 1]], order=1, cval=numpy.mean(data))

            shifted = numpy.moveaxis(shifted, (2, 3), (0, 1))

            shifted_xdata = DataAndMetadata.new_data_and_metadata(shifted, data_descriptor=DataAndMetadata.DataDescriptor(False, 2, 2))

            result = MultiDimensionalProcessing.function_measure_multi_dimensional_shifts(shifted_xdata,
                                                                                          "collection",
                                                                                          reference_index=reference_index)

            self.assertTrue(numpy.allclose(result.data, -1.0 * (shifts - shifts[numpy.unravel_index(reference_index, shifts.shape[:-1])]), atol=0.5))