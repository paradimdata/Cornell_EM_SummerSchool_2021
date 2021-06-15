import gettext
import unittest

import numpy

# local libraries
from nion.swift import Facade
from nion.data import DataAndMetadata
from nion.swift.test import TestContext
from nion.ui import TestUI
from nion.swift import Application
from nion.swift.model import DocumentModel

from nionswift_plugin.nion_experimental_tools import AffineTransformImage

_ = gettext.gettext


Facade.initialize()


def create_memory_profile_context() -> TestContext.MemoryProfileContext:
    return TestContext.MemoryProfileContext()


class TestAffineTransformImage(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=True)
        self.app.workspace_dir = str()

    def tearDown(self):
        pass

    def test_affine_transform_image_for_2d_data(self):
        with create_memory_profile_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.zeros((5, 5))
            data[2:-2, 1:-1] = 1
            xdata = DataAndMetadata.new_data_and_metadata(data)
            api = Facade.get_api("~1.0", "~1.0")
            data_item = api.library.create_data_item_from_data_and_metadata(xdata)
            document_controller.selection.set(0)
            document_controller.selected_display_panel = None  # use the document controller selection
            affine_transform = AffineTransformImage.AffineTransformMenuItem(api)
            affine_transform.menu_item_execute(api.application.document_controllers[0])
            document_controller.periodic()
            # Can't convince the computation to update when changing the graphics, so just check that it got executed
            vector_a = data_item.graphics[0]
            vector_b = data_item.graphics[1]
            # # Rotate by 90 degrees
            vector_a.end = (0.75, 0.5)
            vector_b.end = (0.5, 0.75)
            # # Update computation
            document_controller.periodic()
            DocumentModel.evaluate_data(document_model.computations[0])
            self.assertEqual(len(data_item.graphics), 2)
            self.assertEqual(api.library.data_item_count, 2)
            self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data)))

    def test_affine_transform_image_for_3d_data(self):
        data_descriptors = [DataAndMetadata.DataDescriptor(True, 0, 2), DataAndMetadata.DataDescriptor(False, 1, 2),
                            DataAndMetadata.DataDescriptor(False, 2, 1)]
        for data_descriptor in data_descriptors:
            with self.subTest(data_descriptor=data_descriptor):
                with create_memory_profile_context() as profile_context:
                    document_controller = profile_context.create_document_controller_with_application()
                    document_model = document_controller.document_model
                    data = numpy.zeros((5, 5, 5))
                    if data_descriptor.collection_dimension_count == 2:
                        data[2:-2, 1:-1] = 1
                    else:
                        data[..., 2:-2, 1:-1] = 1
                    xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)
                    api = Facade.get_api("~1.0", "~1.0")
                    data_item = api.library.create_data_item_from_data_and_metadata(xdata)
                    document_controller.selection.set(0)
                    document_controller.selected_display_panel = None  # use the document controller selection
                    affine_transform = AffineTransformImage.AffineTransformMenuItem(api)
                    affine_transform.menu_item_execute(api.application.document_controllers[0])
                    document_controller.periodic()
                    # Can't convince the computation to update when changing the graphics, so just check that it got executed
                    vector_a = data_item.graphics[0]
                    vector_b = data_item.graphics[1]
                    # # Rotate by 90 degrees
                    vector_a.end = (0.75, 0.5)
                    vector_b.end = (0.5, 0.75)
                    # # Update computation
                    document_controller.periodic()
                    DocumentModel.evaluate_data(document_model.computations[0])
                    self.assertEqual(len(data_item.graphics), 2)
                    self.assertEqual(api.library.data_item_count, 2)
                    if data_descriptor.collection_dimension_count == 2:
                        self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data)))
                    else:
                        self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data, axes=(1, 2))))

    def test_affine_transform_image_for_4d_data(self):
        data_descriptors = [DataAndMetadata.DataDescriptor(True, 1, 2), DataAndMetadata.DataDescriptor(False, 2, 2),
                            DataAndMetadata.DataDescriptor(True, 2, 1)]
        for data_descriptor in data_descriptors:
            with self.subTest(data_descriptor=data_descriptor):
                with create_memory_profile_context() as profile_context:
                    document_controller = profile_context.create_document_controller_with_application()
                    document_model = document_controller.document_model
                    data = numpy.zeros((5, 5, 5, 5))
                    if data_descriptor.collection_dimension_count == 2 and not data_descriptor.is_sequence:
                        data[2:-2, 1:-1] = 1
                    elif data_descriptor.collection_dimension_count == 2 and data_descriptor.is_sequence:
                        data[:, 2:-2, 1:-1] = 1
                    else:
                        data[..., 2:-2, 1:-1] = 1
                    xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)
                    api = Facade.get_api("~1.0", "~1.0")
                    data_item = api.library.create_data_item_from_data_and_metadata(xdata)
                    document_controller.selection.set(0)
                    document_controller.selected_display_panel = None  # use the document controller selection
                    affine_transform = AffineTransformImage.AffineTransformMenuItem(api)
                    affine_transform.menu_item_execute(api.application.document_controllers[0])
                    document_controller.periodic()
                    # Can't convince the computation to update when changing the graphics, so just check that it got executed
                    vector_a = data_item.graphics[0]
                    vector_b = data_item.graphics[1]
                    # # Rotate by 90 degrees
                    vector_a.end = (0.75, 0.5)
                    vector_b.end = (0.5, 0.75)
                    # # Update computation
                    document_controller.periodic()
                    DocumentModel.evaluate_data(document_model.computations[0])
                    self.assertEqual(len(data_item.graphics), 2)
                    self.assertEqual(api.library.data_item_count, 2)
                    if data_descriptor.collection_dimension_count == 2 and not data_descriptor.is_sequence:
                        self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data)))
                    elif data_descriptor.collection_dimension_count == 2 and data_descriptor.is_sequence:
                        self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data, axes=(1, 2))))
                    else:
                        self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data, axes=(2, 3))))

    def test_affine_transform_image_for_5d_data(self):
        data_descriptor = DataAndMetadata.DataDescriptor(True, 2, 2)
        with create_memory_profile_context() as profile_context:
            document_controller = profile_context.create_document_controller_with_application()
            document_model = document_controller.document_model
            data = numpy.zeros((2, 5, 5, 5, 5))
            data[:, 2:-2, 1:-1] = 1
            xdata = DataAndMetadata.new_data_and_metadata(data, data_descriptor=data_descriptor)
            api = Facade.get_api("~1.0", "~1.0")
            data_item = api.library.create_data_item_from_data_and_metadata(xdata)
            document_controller.selection.set(0)
            document_controller.selected_display_panel = None  # use the document controller selection
            affine_transform = AffineTransformImage.AffineTransformMenuItem(api)
            affine_transform.menu_item_execute(api.application.document_controllers[0])
            document_controller.periodic()
            # Can't convince the computation to update when changing the graphics, so just check that it got executed
            vector_a = data_item.graphics[0]
            vector_b = data_item.graphics[1]
            # # Rotate by 90 degrees
            vector_a.end = (0.75, 0.5)
            vector_b.end = (0.5, 0.75)
            # # Update computation
            document_controller.periodic()
            DocumentModel.evaluate_data(document_model.computations[0])
            self.assertEqual(len(data_item.graphics), 2)
            self.assertEqual(api.library.data_item_count, 2)
            self.assertTrue(numpy.allclose(document_model.data_items[1].data, numpy.rot90(data, axes=(1, 2))))
