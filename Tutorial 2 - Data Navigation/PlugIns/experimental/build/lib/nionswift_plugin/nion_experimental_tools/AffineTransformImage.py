# system imports
import gettext
import logging
import copy

import numpy

# local libraries
from nion.typeshed import API_1_0 as API
from nion.data import xdata_1_0 as xd
from nion.swift import Facade
from nion.swift.model import Symbolic
from nion.data import DataAndMetadata
from nion.data import Core

_ = gettext.gettext


class AffineTransformImage:
    label = _("Affine Transform Image")
    inputs = {
        "src_data_item": {"label": _("Source")},
        "vector_a": {"label": _("Vector a")},
        "vector_b": {"label": _("Vector b")},
        }
    outputs = {
        "target": {"label": _("Affine Transformed")},
    }
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, src_data_item, vector_a, vector_b, **kwargs) -> None:
        try:
            vector_a = vector_a[0]
            vector_b = vector_b[0]
            vector_1 = (numpy.array(vector_a.end) - numpy.array(vector_a.start)) * 4.0
            vector_2 = (numpy.array(vector_b.end) - numpy.array(vector_b.start)) * 4.0

            matrix = numpy.array(((vector_1[1], vector_2[1]),
                                  (vector_1[0], vector_2[0])))
            xdata = src_data_item.xdata
            if not xdata:
                return
            if (xdata.is_sequence or xdata.is_collection) and xdata.collection_dimension_count != 2:
                navigation_dimensions = 0
                if xdata.is_sequence:
                    navigation_dimensions += 1
                navigation_dimensions += xdata.collection_dimension_count
                navigation_dimensions_shape = xdata.data.shape[:navigation_dimensions]
                result_data = numpy.zeros_like(xdata.data)
                new_coords = Core.calculate_coordinates_for_affine_transform(xdata.data[numpy.unravel_index(0, navigation_dimensions_shape)], matrix)
                for i in range(numpy.prod(navigation_dimensions_shape)):
                    index = numpy.unravel_index(i, navigation_dimensions_shape)
                    transformed = xd.warp(xdata.data[index], new_coords)
                    result_data[index] = transformed.data
                self._affine_transformed_xdata = DataAndMetadata.new_data_and_metadata(result_data,
                                                                                        intensity_calibration=xdata.intensity_calibration,
                                                                                        dimensional_calibrations=xdata.dimensional_calibrations,
                                                                                        data_descriptor=xdata.data_descriptor)
            elif xdata.collection_dimension_count == 2: # Assume we want to distort allong the collection dimension in this case
                navigation_dimensions_shape = tuple()
                if xdata.is_sequence:
                    navigation_dimensions_shape = (xdata.data_shape[0],) + xdata.data_shape[3:]
                else:
                    navigation_dimensions_shape = xdata.data_shape[2:]

                if xdata.is_data_rgb_type:
                    navigation_dimensions_shape = navigation_dimensions_shape[:-1]
                index = numpy.unravel_index(0, navigation_dimensions_shape)
                index = ((index[0],) if xdata.is_sequence else tuple()) + (slice(None), slice(None)) + index[-xdata.datum_dimension_count:]
                new_coords = Core.calculate_coordinates_for_affine_transform(xdata.data[index], matrix)
                result_data = numpy.zeros_like(xdata.data)
                for i in range(numpy.prod(navigation_dimensions_shape)):
                    index = numpy.unravel_index(i, navigation_dimensions_shape)
                    index = ((index[0],) if xdata.is_sequence else tuple()) + (slice(None), slice(None)) + index[-xdata.datum_dimension_count:]
                    transformed = xd.warp(xdata.data[index], new_coords)
                    result_data[index] = transformed.data
                self._affine_transformed_xdata = DataAndMetadata.new_data_and_metadata(result_data,
                                                                                        intensity_calibration=xdata.intensity_calibration,
                                                                                        dimensional_calibrations=xdata.dimensional_calibrations,
                                                                                        data_descriptor=xdata.data_descriptor)
            else:
                self._affine_transformed_xdata = xd.affine_transform(xdata, matrix)
            metadata = self._affine_transformed_xdata.metadata
            metadata["nion.affine_transform_image.transformation_matrix"] = matrix.tolist()
            self._affine_transformed_xdata._set_metadata(metadata)
        except:
            import traceback
            traceback.print_exc()
            raise

    def commit(self):
        self.computation.set_referenced_xdata("target", self._affine_transformed_xdata)


class AffineTransformMenuItem:
    menu_id = "_processing_menu"
    menu_item_name = _("Affine Transform image")

    def __init__(self, api):
        self.__api = api

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        data_item = window.target_data_item

        if not data_item:
            return

        vector_a = data_item.add_line_region(0.5, 0.5, 0.5, 0.75)
        vector_a.label = "Vector a"
        vector_a._graphic.end_arrow_enabled = True
        vector_b = data_item.add_line_region(0.5, 0.5, 0.75, 0.5)
        vector_b.label = "Vector b"
        vector_b._graphic.end_arrow_enabled = True

        result_data_item = self.__api.library.create_data_item_from_data_and_metadata(DataAndMetadata.new_data_and_metadata(numpy.zeros_like(data_item.data), data_descriptor=data_item.xdata.data_descriptor), title="Affine Transformed {}".format(data_item.title))
        self.__api.library.create_computation("nion.affine_transform_image",
                                              inputs={"src_data_item": data_item, "vector_a": [vector_a], "vector_b": [vector_b]},
                                              outputs={"target": result_data_item})
        window.display_data_item(result_data_item)

class CopyAffineTransformMenuItem:
    menu_id = "_processing_menu"
    menu_item_name = _("Copy Affine Transformation")

    def __init__(self, api):
        self.__api = api

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        document_controller = window._document_controller
        selected_display_items = document_controller.selected_display_items
        data_items = list()
        src_vector_a = None
        src_vector_b = None
        for i, display_item in enumerate(selected_display_items):
            data_item = display_item.data_items[0] if display_item and len(display_item.data_items) > 0 else None
            if data_item and src_vector_a is None:
                for computation in self.__api.library._document_model.computations:
                    if computation.processing_id == "nion.affine_transform_image":
                        if computation.get_input("src_data_item") == data_item:
                            src_vector_a = computation.get_input("vector_a")
                            if src_vector_a is not None:
                                src_vector_a = self.__api._new_api_object(src_vector_a[0])
                            src_vector_b = computation.get_input("vector_b")
                            if src_vector_b is not None:
                                src_vector_b = self.__api._new_api_object(src_vector_b[0])
                            break
                else:
                    data_items.append(self.__api._new_api_object(data_item))

            elif data_item:
                data_items.append(self.__api._new_api_object(data_item))
        if src_vector_a is None:
            return

        for data_item in data_items:
            vector_a = data_item.add_line_region(src_vector_a.start[0], src_vector_a.start[1], src_vector_a.end[0], src_vector_a.end[1])
            vector_a.label = "Vector a"
            vector_a._graphic.end_arrow_enabled = True
            vector_b = data_item.add_line_region(src_vector_b.start[0], src_vector_b.start[1], src_vector_b.end[0], src_vector_b.end[1])
            vector_b.label = "Vector b"
            vector_b._graphic.end_arrow_enabled = True

            result_data_item = self.__api.library.create_data_item_from_data_and_metadata(DataAndMetadata.new_data_and_metadata(numpy.zeros_like(data_item.data), data_descriptor=data_item.xdata.data_descriptor), title="Affine Transformed {}".format(data_item.title))
            self.__api.library.create_computation("nion.affine_transform_image",
                                                  inputs={"src_data_item": data_item, "vector_a": [vector_a], "vector_b": [vector_b]},
                                                  outputs={"target": result_data_item})
            window.display_data_item(result_data_item)


class AffineTransformExtension:
    extension_id = "nion.extension.affine_transform"

    def __init__(self, api_broker):
        api = api_broker.get_api(version="1", ui_version="1")

        self.__affine_transform_menu_item_ref = api.create_menu_item(AffineTransformMenuItem(api))
        self.__copy_affine_transform_menu_item_ref = api.create_menu_item(CopyAffineTransformMenuItem(api))

    def close(self):
        self.__affine_transform_menu_item_ref.close()
        self.__copy_affine_transform_menu_item_ref.close()


Symbolic.register_computation_type("nion.affine_transform_image", AffineTransformImage)