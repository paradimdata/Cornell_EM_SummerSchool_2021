import typing
import gettext
import copy
import numpy
import scipy.ndimage
import threading
import time

from nion.data import Core
from nion.data import DataAndMetadata
from nion.data import Calibration
from nion.swift.model import Symbolic
from nion.swift.model import Schema
from nion.swift.model import DataStructure
from nion.swift.model import DataItem
from nion.typeshed import API_1_0 as API

_ = gettext.gettext


class IntegrateAlongAxis:
    label = _("Integrate")
    inputs = {"input_data_item": {"label": _("Input data item")},
              "integration_axes": {"label": _("Integrate along this axis"), "entity_id": "axis_choice"},
              "integration_graphic": {"label": _("Integration mask")},
              }
    outputs = {"integrated": {"label": _("Integrated")},
               }

    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, input_data_item: API.DataItem, integration_axes: str, integration_graphic: typing.Optional[API.Graphic]=None):
        input_xdata: DataAndMetadata.DataAndMetadata = input_data_item.xdata
        integration_axes = integration_axes._data_structure.entity.entity_type.entity_id
        if integration_axes == "collection":
            assert input_xdata.is_collection
            integration_axis_indices = list(input_xdata.collection_dimension_indexes)
            integration_axis_shape = input_xdata.collection_dimension_shape
            result_data_descriptor = DataAndMetadata.DataDescriptor(input_xdata.is_sequence, 0, input_xdata.datum_dimension_count)
        elif integration_axes == "sequence":
            assert input_xdata.is_sequence
            integration_axis_indices = [input_xdata.sequence_dimension_index]
            integration_axis_shape = input_xdata.sequence_dimension_shape
            result_data_descriptor = DataAndMetadata.DataDescriptor(False, input_xdata.collection_dimension_count, input_xdata.datum_dimension_count)
        else:
            integration_axis_indices = list(input_xdata.datum_dimension_indexes)
            integration_axis_shape = input_xdata.datum_dimension_shape
            # 0-D data is not allowed in Swift, so we need to make the collection or the sequence axis the data axis
            # Use the collection axis preferably and only when the data is not a collection use the sequence axis
            # If the user integrated a single image we get a single number. We also make this 1D data to prevent errors
            if input_xdata.is_collection:
                result_data_descriptor = DataAndMetadata.DataDescriptor(input_xdata.is_sequence, 0, input_xdata.collection_dimension_count)
            else:
                result_data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 1)

        navigation_shape = []
        navigation_axis_indices = []
        for i in range(len(input_xdata.data_shape)):
            if not i in integration_axis_indices:
                navigation_shape.append(input_xdata.data_shape[i])
                navigation_axis_indices.append(i)

        data_str = ''
        mask_str = ''
        navigation_str = ''
        for i in range(len(input_xdata.data_shape)):
            char = chr(i + 97)
            data_str += char
            if i in integration_axis_indices:
                mask_str += char
            else:
                navigation_str += char
        # chr(97) == 'a' so we get letters in alphabetic order here (a, b, c, d, ...)
        sum_str = ''.join([chr(i + 97) for i in range(len(integration_axis_shape))])
        operands = [input_xdata.data]
        if integration_graphic is not None:
            mask = integration_graphic.mask_xdata_with_shape(integration_axis_shape)
            operands.append(mask)
            sum_str = data_str + ',' + mask_str
        else:
            sum_str = data_str + '->' + navigation_str

        result_data = numpy.einsum(sum_str, *operands)

        # result_data = numpy.empty(navigation_shape, dtype=input_xdata.data_dtype)

        # last_reported = time.time()
        # n_images = numpy.prod(navigation_shape, dtype=numpy.int64)
        # load_time = 0
        # process_time = 0
        # starttime = time.time()
        # for i in range(n_images):
        #     coords = numpy.unravel_index(i, navigation_shape)
        #     data_coords = coords[:integration_axis_indices[0]] + (...,) + coords[integration_axis_indices[0]:]
        #     t0 = time.perf_counter()
        #     operands[0] = input_xdata.data[data_coords]
        #     t1 = time.perf_counter()
        #     result_data[coords] = numpy.einsum(sum_str, *operands)
        #     t2 = time.perf_counter()
        #     load_time += t1 - t0
        #     process_time += t2 - t1
        #     now = time.time()
        #     if now - last_reported > 3.0:
        #         last_reported = now
        #         print(f"Processed {i}/{n_images} data points ({i/(now - starttime):.0f} dpps). Spent {load_time:.1f} s loading data and {process_time:.1f} s processing data so far.")

        result_dimensional_calibrations = []
        for i in range(len(input_xdata.data_shape)):
            if not i in integration_axis_indices:
                result_dimensional_calibrations.append(input_xdata.dimensional_calibrations[i])

        self.__result_xdata = DataAndMetadata.new_data_and_metadata(numpy.atleast_1d(result_data),
                                                                    intensity_calibration=input_xdata.intensity_calibration,
                                                                    dimensional_calibrations=result_dimensional_calibrations,
                                                                    data_descriptor=result_data_descriptor)

    def commit(self):
        self.computation.set_referenced_xdata("integrated", self.__result_xdata)



def function_measure_multi_dimensional_shifts(xdata: DataAndMetadata.DataAndMetadata,
                                              shift_axis: str,
                                              reference_index: typing.Union[None, int, typing.Sequence[int]]=None,
                                              bounds: typing.Optional[typing.Sequence[int]]=None) -> numpy.ndarray:
    if shift_axis == "collection":
        assert xdata.is_collection
        if xdata.collection_dimension_count == 2:
            shifts_ndim = 1
        else:
            shifts_ndim = 0
        shift_axis_indices = list(xdata.collection_dimension_indexes)
    elif shift_axis == "sequence":
        assert xdata.is_sequence
        shifts_ndim = 0
        shift_axis_indices = [xdata.sequence_dimension_index]
    elif shift_axis == "data":
        if xdata.datum_dimension_count == 2:
            shifts_ndim = 1
        else:
            shifts_ndim = 0
        shift_axis_indices = list(xdata.datum_dimension_indexes)
    else:
        raise ValueError(f"Unknown shift axis: '{shift_axis}'.")

    iteration_shape = list()
    dimensional_calibrations = list()
    intensity_calibration = None
    for i in range(len(xdata.data_shape)):
        if not i in shift_axis_indices:
            iteration_shape.append(xdata.data_shape[i])
            dimensional_calibrations.append(xdata.dimensional_calibrations[i])
        else:
            intensity_calibration = xdata.dimensional_calibrations[i]
    iteration_shape = tuple(iteration_shape)

    if shifts_ndim == 1:
        result_shape = iteration_shape + (2,)
        dimensional_calibrations.append(Calibration.Calibration())
        if bounds is not None:
            assert numpy.ndim(bounds) == 2
            shape = (xdata.data_shape[shift_axis_indices[0]], xdata.data_shape[shift_axis_indices[1]])
            register_slice = (slice(max(0, int(round(bounds[0][0] * shape[0]))), min(int(round((bounds[0][0] + bounds[1][0]) * shape[0])), shape[0])),
                              slice(max(0, int(round(bounds[0][1] * shape[1]))), min(int(round((bounds[0][1] + bounds[1][1]) * shape[1])), shape[1])))
        else:
            register_slice = (slice(0, None), slice(0, None))
    else:
        result_shape = iteration_shape + (1,)
        if bounds is not None:
            assert numpy.ndim(bounds) == 1
            shape = (xdata.data_shape[shift_axis_indices[0]],)
            register_slice = slice(max(0, int(round(bounds[0] * shape[0]))), min(int(round(bounds[1] * shape[0])), shape[0]))
        else:
            register_slice = slice(0, None)

    if reference_index is not None:
        if numpy.isscalar(reference_index):
            coords = numpy.unravel_index(reference_index, iteration_shape)
        else:
            coords = reference_index
        data_coords = coords[:shift_axis_indices[0]] + (...,) + coords[shift_axis_indices[0]:]
        reference_data = xdata.data[data_coords]

    shifts = numpy.zeros(result_shape, dtype=numpy.float32)

    start_index = 0 if reference_index is not None else 1

    for i in range(start_index, numpy.prod(iteration_shape, dtype=numpy.int64)):
        coords = numpy.unravel_index(i, iteration_shape)
        data_coords = coords[:shift_axis_indices[0]] + (...,) + coords[shift_axis_indices[0]:]
        if reference_index is None:
            coords_ref = numpy.unravel_index(i - 1, iteration_shape)
            data_coords_ref = coords_ref[:shift_axis_indices[0]] + (...,) + coords_ref[shift_axis_indices[0]:]
            reference_data = xdata.data[data_coords_ref]
        shifts[coords] = Core.function_register_template(reference_data[register_slice], xdata.data[data_coords][register_slice])[1]

    shifts = numpy.squeeze(shifts)

    if reference_index is None:
        shifts = numpy.cumsum(shifts, axis=0)

    return DataAndMetadata.new_data_and_metadata(shifts,
                                                 intensity_calibration=intensity_calibration,
                                                 dimensional_calibrations=dimensional_calibrations)


def function_apply_multi_dimensional_shifts(xdata: DataAndMetadata.DataAndMetadata,
                                            shifts: numpy.ndarray,
                                            shift_axis: str,
                                            out: typing.Optional[DataAndMetadata.DataAndMetadata] = None):
    if shift_axis == "collection":
        assert xdata.is_collection
        if xdata.collection_dimension_count == 2:
            assert shifts.shape[-1] == 2
            shifts_shape = shifts.shape[:-1]
        else:
            shifts_shape = shifts.shape
        shift_axis_indices = list(xdata.collection_dimension_indexes)
    elif shift_axis == "sequence":
        assert xdata.is_sequence
        shifts_shape = shifts.shape
        shift_axis_indices = [xdata.sequence_dimension_index]
    elif shift_axis == "data":
        if xdata.datum_dimension_count == 2:
            assert shifts.shape[-1] == 2
            shifts_shape = shifts.shape[:-1]
        else:
            shifts_shape = shifts.shape
        shift_axis_indices = list(xdata.datum_dimension_indexes)
    else:
        raise ValueError(f"Unknown shift axis: '{shift_axis}'.")

    # Find the axes that we do not want to shift (== iteration shape)
    iteration_shape = list()
    for i in range(len(xdata.data_shape)):
        if not i in shift_axis_indices:
            iteration_shape.append(xdata.data_shape[i])
    iteration_shape = tuple(iteration_shape)

    # Now we need to find matching axes between the iteration shape and the provided shifts. We can then iterate over
    # these matching axis and apply the shifts.
    for i in range(len(iteration_shape) - len(shifts_shape) + 1):
        if iteration_shape[i:i+len(shifts_shape)] == shifts_shape:
            shifts_start_axis = i
            shifts_end_axis = i + len(shifts_shape)
            break
    else:
        raise ValueError("Did not find any axis matching the shifts shape.")

    # Now drop all iteration axes after the last shift axis. This will greatly improve speed because we don't have
    # to iterate and shift each individual element but can work in larger sub-arrays. It will also be beneficial for
    # working with chunked hdf5 files because we usually have our chunks along the last axes.
    squeezed_iteration_shape = iteration_shape[:shifts_end_axis]
    # Chunking it up finer (still aligned with chunks on disk) does not make it faster (actually slower by about a factor
    # of 3). This might change with a storage handler that allows multi-threaded access but for now with h5py we don't
    # want to use this.
    # squeezed_iteration_shape = iteration_shape[:max(shifts_end_axis, shift_axis_indices[0])]

    if out is None:
        result = numpy.empty(xdata.data_shape, dtype=xdata.data_dtype)
    else:
        result = out.data

    # for i in range(numpy.prod(squeezed_iteration_shape, dtype=numpy.int64)):
    #     coords = numpy.unravel_index(i, squeezed_iteration_shape)
    #     for i, ind in enumerate(shift_axis_indices):
    #         shifts_array[ind - len(squeezed_iteration_shape)] = shifts[coords][i]
    #     result[coords] = scipy.ndimage.shift(xdata.data[coords], shifts_array, order=1)

    navigation_len = numpy.prod(squeezed_iteration_shape, dtype=numpy.int64)
    sections = list(range(0, navigation_len, max(1, navigation_len//8)))
    sections.append(navigation_len)
    barrier = threading.Barrier(len(sections))

    def run_on_thread(range_):
        try:
            shifts_array = numpy.zeros(len(shift_axis_indices) + (len(iteration_shape) - len(squeezed_iteration_shape)))
            for i in range_:
                coords = numpy.unravel_index(i, squeezed_iteration_shape)
                for j, ind in enumerate(shift_axis_indices):
                    shift_coords = coords[:shifts_end_axis]
                    shifts_array[ind - len(squeezed_iteration_shape)] = shifts[shift_coords][j]
                # if i % max((range_.stop - range_.start) // 4, 1) == 0:
                #     print(f'Working on slice {coords}: shifting by {shifts_array}')
                result[coords] = scipy.ndimage.shift(xdata.data[coords], shifts_array, order=1)
        finally:
            barrier.wait()

    for i in range(len(sections) - 1):
        threading.Thread(target=run_on_thread, args=(range(sections[i], sections[i+1]),)).start()
    barrier.wait()

    if out is None:
        return DataAndMetadata.new_data_and_metadata(result,
                                                     intensity_calibration=xdata.intensity_calibration,
                                                     dimensional_calibrations=xdata.dimensional_calibrations,
                                                     metadata=xdata.metadata,
                                                     data_descriptor=xdata.data_descriptor)


class MeasureShifts:
    label = _("Measure shifts")
    inputs = {"input_data_item": {"label": _("Input data item")},
              "shift_axis": {"label": _("Measure shift along this axis"), "entity_id": "axis_choice"},
              "reference_index": {"label": _("Reference index for shifts")},
              "bounds_graphic": {"label": _("Shift bounds")},
              }
    outputs = {"shifts": {"label": _("Shifts")},
               }

    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, input_data_item: API.DataItem, shift_axis: str, reference_index: typing.Union[None, int, typing.Sequence[int]]=None, bounds_graphic: typing.Optional[API.Graphic]=None):
        input_xdata = input_data_item.xdata
        bounds = None
        if bounds_graphic is not None:
            if bounds_graphic.graphic_type == "interval-graphic":
                bounds = bounds_graphic.interval
            else:
                bounds = bounds_graphic.bounds
        shift_axis = shift_axis._data_structure.entity.entity_type.entity_id
        self.__shifts_xdata = function_measure_multi_dimensional_shifts(input_xdata, shift_axis, reference_index=reference_index, bounds=bounds)

    def commit(self):
        self.computation.set_referenced_xdata("shifts", self.__shifts_xdata)


class MeasureShiftsMenuItemDelegate:
    def __init__(self, api: API.API):
        self.__api = api
        self.menu_id = "multi_dimensional_processing_menu"
        self.menu_name = _("Multi-Dimensional Processing")
        self.menu_before_id = "window_menu"
        self.menu_item_name = _("Measure shifts")

    def menu_item_execute(self, window: API.DocumentWindow):
        selected_data_item = window.target_data_item

        if not selected_data_item or not selected_data_item.xdata:
            return

        bounds_graphic = None
        if selected_data_item.display.selected_graphics:
            for graphic in selected_data_item.display.selected_graphics:
                if graphic.graphic_type in {"rect-graphic", "interval-graphic"}:
                    bounds_graphic = graphic

        # If we have a bound graphic we probably want to align the displayed dimensions
        if bounds_graphic:
            # For collections with 1D data we see the collection dimensions
            if selected_data_item.xdata.is_collection and selected_data_item.xdata.datum_dimension_count == 1:
                shift_axis = 'collection'
            # Otherwise we see the data dimensions
            else:
                shift_axis = 'data'
        # If not, use some generic rules
        else:
            shift_axis = 'data'

            if selected_data_item.xdata.is_collection and selected_data_item.xdata.datum_dimension_count == 1:
                shift_axis = 'collection'


        # Make a result data item with 3 dimensions to ensure we get a large_format data item
        result_data_item = self.__api.library.create_data_item_from_data(numpy.zeros((1,1,1)), title="Shifts of {}".format(selected_data_item.title))

        shift_axis_structure = DataStructure.DataStructure(structure_type=shift_axis)
        self.__api.library._document_model.append_data_structure(shift_axis_structure)
        shift_axis_structure.source = result_data_item._data_item

        inputs = {"input_data_item": selected_data_item,
                  "shift_axis": self.__api._new_api_object(shift_axis_structure),
                  "reference_index": 0,
                  }
        if bounds_graphic:
            inputs["bounds_graphic"] = bounds_graphic

        computation = self.__api.library.create_computation("nion.measure_shifts",
                                                            inputs=inputs,
                                                            outputs={"shifts": result_data_item})
        computation._computation.source = result_data_item._data_item
        window.display_data_item(result_data_item)


class ApplyShifts:
    label = _("Apply shifts")
    inputs = {"input_data_item": {"label": _("Input data item")},
              "shifts_data_item": {"label": _("Shifts data item")},
              "shift_axis": {"label": _("Apply shift along this axis"), "entity_id": "axis_choice"},
              }
    outputs = {"shifted": {"label": _("Shifted")},
               }

    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, input_data_item: API.DataItem, shifts_data_item: API.DataItem, shift_axis: str):
        input_xdata = input_data_item.xdata
        shifts  = shifts_data_item.data
        shift_axis = shift_axis._data_structure.entity.entity_type.entity_id
        # Like this we directly write to the underlying storage and don't have to cache everything in memory first
        result_data_item = self.computation.get_result('shifted')
        function_apply_multi_dimensional_shifts(input_xdata, shifts, shift_axis, out=result_data_item.xdata)
        # self.__result_xdata = function_apply_multi_dimensional_shifts(input_xdata, shifts, shift_axis)

    def commit(self):
        # self.computation.set_referenced_xdata("shifted", self.__result_xdata)
        # self.__result_xdata = None
        # Still call "set_referenced_xdata" to notify Swift that the data has been updated.
        self.computation.set_referenced_xdata("shifted", self.computation.get_result("shifted").xdata)


class ApplyShiftsMenuItemDelegate:
    def __init__(self, api: API.API):
        self.__api = api
        self.menu_id = "multi_dimensional_processing_menu"
        self.menu_name = _("Multi-Dimensional Processing")
        self.menu_before_id = "window_menu"
        self.menu_item_name = _("Apply shifts")

    def menu_item_execute(self, window: API.DocumentWindow):
        selected_display_items = window._document_controller._get_two_data_sources()
        error_msg = "Select a multi-dimensional data item and another one that contains shifts that can be broadcast to the shape of the first one."
        assert selected_display_items[0][0] is not None, error_msg
        assert selected_display_items[1][0] is not None, error_msg
        assert selected_display_items[0][0].data_item is not None, error_msg
        assert selected_display_items[1][0].data_item is not None, error_msg
        assert selected_display_items[0][0].data_item.xdata is not None, error_msg
        assert selected_display_items[1][0].data_item.xdata is not None, error_msg

        di_1 = selected_display_items[0][0].data_item
        di_2 = selected_display_items[1][0].data_item

        if len(di_1.data.shape) < len(di_2.data.shape):
            shifts_di = self.__api._new_api_object(di_1)
            input_di = self.__api._new_api_object(di_2)
        elif len(di_2.data.shape) < len(di_1.data.shape):
            shifts_di = self.__api._new_api_object(di_2)
            input_di = self.__api._new_api_object(di_1)
        else:
            raise ValueError(error_msg)

        shifts_shape = shifts_di.data.shape
        data_shape = input_di.data.shape
        for i in range(len(data_shape) - len(shifts_shape) + 1):
            if data_shape[i:i+len(shifts_shape)] == shifts_shape:
                shifts_start_axis = i
                shifts_end_axis = i + len(shifts_shape)
                break
            elif data_shape[i:i+len(shifts_shape)-1] == shifts_shape[:-1] and shifts_shape[-1] == 2:
                shifts_start_axis = i
                shifts_end_axis = i + len(shifts_shape) - 1
                break
        else:
            raise ValueError("Did not find any axis matching the shifts shape.")

        shifts_indexes = range(shifts_start_axis, shifts_end_axis)
        shift_axis_points = {"collection": 0, "sequence": 0, "data": 0}
        if input_di.xdata.is_collection:
            collection_dimension_indexes = input_di.xdata.collection_dimension_indexes
            cond = False
            for ind in collection_dimension_indexes:
                if ind in shifts_indexes:
                    cond = True
            if not cond and (len(collection_dimension_indexes) == 1 or len(collection_dimension_indexes) == shifts_shape[-1]):
                shift_axis_points["collection"] += 1

        if input_di.xdata.is_sequence:
            sequence_dimension_index = input_di.xdata.sequence_dimension_index
            if not sequence_dimension_index in shifts_indexes:
                shift_axis_points["sequence"] += 1

        datum_dimension_indexes = input_di.xdata.datum_dimension_indexes
        cond = False
        for ind in datum_dimension_indexes:
            if ind in shifts_indexes:
                cond = True
        if not cond and (len(datum_dimension_indexes) == 1 or len(datum_dimension_indexes) == shifts_shape[-1]):
            shift_axis_points["data"] += 1

        if shift_axis_points["data"] > 0:
            shift_axis = "data"
        elif shift_axis_points["collection"] > 0:
            shift_axis = "collection"
        elif shift_axis_points["sequence"] > 0:
            shift_axis = "sequence"
        else:
            shift_axis = "data"

        data_item = DataItem.DataItem(large_format=True)
        data_item.title="Shifted {}".format(input_di.title)
        window._document_controller.document_model.append_data_item(data_item)
        data_item.reserve_data(data_shape=input_di.xdata.data_shape, data_dtype=input_di.xdata.data_dtype, data_descriptor=input_di.xdata.data_descriptor)
        data_item.dimensional_calibrations = input_di.xdata.dimensional_calibrations
        data_item.intensity_calibration = input_di.xdata.intensity_calibration
        data_item.metadata = copy.deepcopy(input_di.xdata.metadata)
        result_data_item = self.__api._new_api_object(data_item)

        shift_axis_structure = DataStructure.DataStructure(structure_type=shift_axis)
        self.__api.library._document_model.append_data_structure(shift_axis_structure)
        shift_axis_structure.source = result_data_item._data_item

        inputs = {"input_data_item": input_di,
                  "shifts_data_item": shifts_di,
                  "shift_axis": self.__api._new_api_object(shift_axis_structure)
                  }

        computation = self.__api.library.create_computation("nion.apply_shifts",
                                                            inputs=inputs,
                                                            outputs={"shifted": result_data_item})
        computation._computation.source = result_data_item._data_item
        window.display_data_item(result_data_item)


class IntegrateAlongAxisMenuItemDelegate:
    def __init__(self, api: API.API):
        self.__api = api
        self.menu_id = "multi_dimensional_processing_menu"
        self.menu_name = _("Multi-Dimensional Processing")
        self.menu_before_id = "window_menu"
        self.menu_item_name = _("Integrate axis")

    def menu_item_execute(self, window: API.DocumentWindow):
        selected_data_item = window.target_data_item

        if not selected_data_item or not selected_data_item.xdata:
            return

        integrate_graphic = None
        if selected_data_item.display.selected_graphics:
            integrate_graphic = selected_data_item.display.selected_graphics[0]

        # If we have an integrate graphic we probably want to integrate the displayed dimensions
        if integrate_graphic:
            # For collections with 1D data we see the collection dimensions
            if selected_data_item.xdata.is_collection and selected_data_item.xdata.datum_dimension_count == 1:
                integration_axes = "collection"
            # Otherwise we see the data dimensions
            else:
                integration_axes = "data"
        # If not, use some generic rules
        else:
            if selected_data_item.xdata.is_sequence:
                integration_axes = "sequence"
            elif selected_data_item.xdata.is_collection and selected_data_item.xdata.datum_dimension_count == 1:
                integration_axes = "collection"
            else:
                integration_axes = "data"

        # Make a result data item with 3 dimensions to ensure we get a large_format data item
        result_data_item = self.__api.library.create_data_item_from_data(numpy.zeros((1,1,1)), title="Integrated {}".format(selected_data_item.title))

        integration_axes_structure = DataStructure.DataStructure(structure_type=integration_axes)
        self.__api.library._document_model.append_data_structure(integration_axes_structure)
        integration_axes_structure.source = result_data_item._data_item

        inputs = {"input_data_item": selected_data_item,
                  "integration_axes": self.__api._new_api_object(integration_axes_structure),
                  }
        if integrate_graphic:
            inputs["integration_graphic"] = integrate_graphic

        computation = self.__api.library.create_computation("nion.integrate_along_axis",
                                                            inputs=inputs,
                                                            outputs={"integrated": result_data_item})
        computation._computation.source = result_data_item._data_item
        window.display_data_item(result_data_item)


class Crop:
    label = _("Crop")
    inputs = {"input_data_item": {"label": _("Input data item")},
              "crop_axis": {"label": _("Crop along this axis"), "entity_id": "axis_choice"},
              "crop_graphic": {"label": _("Crop bounds")},
              "crop_bounds_left": {"label": _("Crop bound left")},
              "crop_bounds_right": {"label": _("Crop bound right")},
              "crop_bounds_top": {"label": _("Crop bound top")},
              "crop_bounds_bottom": {"label": _("Crop bound bottom")}}
    outputs = {"cropped": {"label": _("Cropped")}}

    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, input_data_item: API.DataItem, crop_axis: str, crop_graphic: typing.Optional[API.Graphic]=None, **kwargs):
        input_xdata: DataAndMetadata.DataAndMetadata = input_data_item.xdata
        crop_axis = crop_axis._data_structure.entity.entity_type.entity_id
        if crop_axis == "collection":
            assert input_xdata.is_collection
            crop_axis_indices = list(input_xdata.collection_dimension_indexes)
        elif crop_axis == "sequence":
            assert input_xdata.is_sequence
            crop_axis_indices = [input_xdata.sequence_dimension_index]
        else:
            crop_axis_indices = list(input_xdata.datum_dimension_indexes)

        if crop_graphic is not None:
            if len(crop_axis_indices) == 2:
                bounds = crop_graphic.bounds
                assert numpy.ndim(bounds) == 2
                crop_bounds_left = bounds[0][1] * input_xdata.data_shape[crop_axis_indices[1]]
                crop_bounds_right = (bounds[0][1] + bounds[1][1]) * input_xdata.data_shape[crop_axis_indices[1]]
                crop_bounds_top = bounds[0][0] * input_xdata.data_shape[crop_axis_indices[0]]
                crop_bounds_bottom = (bounds[0][0] + bounds[1][0]) * input_xdata.data_shape[crop_axis_indices[0]]
            else:
                bounds = crop_graphic.interval
                assert numpy.ndim(bounds) == 1
                crop_bounds_left = bounds[0] * input_xdata.data_shape[crop_axis_indices[0]]
                crop_bounds_right = bounds[1] * input_xdata.data_shape[crop_axis_indices[0]]
        else:
            crop_bounds_left = kwargs.get("crop_bounds_left")
            crop_bounds_right = kwargs.get("crop_bounds_right")
            crop_bounds_top = kwargs.get("crop_bounds_top")
            crop_bounds_bottom = kwargs.get("crop_bounds_bottom")

        if len(crop_axis_indices) == 2:
            crop_bounds_left = max(0, crop_bounds_left)
            crop_bounds_top = max(0, crop_bounds_top)
            if crop_bounds_right == -1:
                crop_bounds_right = None
            else:
                crop_bounds_right = min(crop_bounds_right, input_xdata.data_shape[crop_axis_indices[1]])
            if crop_bounds_bottom == -1:
                crop_bounds_bottom = None
            else:
                crop_bounds_bottom = min(crop_bounds_bottom, input_xdata.data_shape[crop_axis_indices[0]])
        else:
            crop_bounds_left = max(0, crop_bounds_left)
            if crop_bounds_right == -1:
                crop_bounds_right = None
            else:
                crop_bounds_right = min(crop_bounds_right, input_xdata.data_shape[crop_axis_indices[0]])

        crop_slices = tuple()
        for i in range(len(input_xdata.data_shape)):
            if len(crop_axis_indices) == 1 and i == crop_axis_indices[0]:
                crop_slices += (slice(crop_bounds_left, crop_bounds_right),)
            elif len(crop_axis_indices) == 2 and i == crop_axis_indices[0]:
                crop_slices += (slice(crop_bounds_top, crop_bounds_bottom),)
            elif len(crop_axis_indices) == 2 and i == crop_axis_indices[1]:
                crop_slices += (slice(crop_bounds_left, crop_bounds_right),)
            else:
                crop_slices += (slice(None),)

        self.__result_xdata = input_xdata[crop_slices]

    def commit(self):
        self.computation.set_referenced_xdata("cropped", self.__result_xdata)


class CropMenuItemDelegate:
    def __init__(self, api: API.API):
        self.__api = api
        self.menu_id = "multi_dimensional_processing_menu"
        self.menu_name = _("Multi-Dimensional Processing")
        self.menu_before_id = "window_menu"
        self.menu_item_name = _("Crop")

    def menu_item_execute(self, window: API.DocumentWindow):
        selected_data_item = window.target_data_item

        if not selected_data_item or not selected_data_item.xdata:
            return

        crop_graphic = None
        if selected_data_item.display.selected_graphics:
            for graphic in selected_data_item.display.selected_graphics:
                if graphic.graphic_type in {"rect-graphic", "interval-graphic"}:
                    crop_graphic = graphic
                    break

        # If we have a crop graphic we probably want to integrate the displayed dimensions
        if crop_graphic:
            # For collections with 1D data we see the collection dimensions
            if selected_data_item.xdata.is_collection and selected_data_item.xdata.datum_dimension_count == 1:
                crop_axes = "collection"
            # Otherwise we see the data dimensions
            else:
                crop_axes = "data"
        # If not, use some generic rules
        else:
            if selected_data_item.xdata.is_collection and selected_data_item.xdata.datum_dimension_count == 1:
                crop_axes = "collection"
            else:
                crop_axes = "data"

        # Make a result data item with 3 dimensions to ensure we get a large_format data item
        result_data_item = self.__api.library.create_data_item_from_data(numpy.zeros((1,1,1)), title="Cropped {}".format(selected_data_item.title))

        crop_axes_structure = DataStructure.DataStructure(structure_type=crop_axes)
        self.__api.library._document_model.append_data_structure(crop_axes_structure)
        crop_axes_structure.source = result_data_item._data_item

        inputs = {"input_data_item": selected_data_item,
                  "crop_axis": self.__api._new_api_object(crop_axes_structure),
                  }
        if crop_graphic:
            inputs["crop_graphic"] = crop_graphic
        else:
            inputs["crop_bounds_left"] = 0
            inputs["crop_bounds_right"] = -1
            inputs["crop_bounds_top"] = 0
            inputs["crop_bounds_bottom"] = -1

        computation = self.__api.library.create_computation("nion.crop_multi_dimensional",
                                                            inputs=inputs,
                                                            outputs={"cropped": result_data_item})
        computation._computation.source = result_data_item._data_item
        window.display_data_item(result_data_item)


class MultiDimensionalProcessingExtension:

    extension_id = "nion.experimental.multi_dimensional_processing"

    def __init__(self, api_broker):
        api = api_broker.get_api(version="~1.0")
        self.__integrate_menu_item_ref = api.create_menu_item(IntegrateAlongAxisMenuItemDelegate(api))
        self.__measure_shifts_menu_item_ref = api.create_menu_item(MeasureShiftsMenuItemDelegate(api))
        self.__apply_shifts_menu_item_ref = api.create_menu_item(ApplyShiftsMenuItemDelegate(api))
        self.__crop_menu_item_ref = api.create_menu_item(CropMenuItemDelegate(api))

    def close(self):
        self.__integrate_menu_item_ref.close()
        self.__integrate_menu_item_ref = None
        self.__measure_shifts_menu_item_ref.close()
        self.__measure_shifts_menu_item_ref = None
        self.__apply_shifts_menu_item_ref.close()
        self.__apply_shifts_menu_item_ref = None
        self.__crop_menu_item_ref.close()
        self.__crop_menu_item_ref = None


Symbolic.register_computation_type("nion.integrate_along_axis", IntegrateAlongAxis)
Symbolic.register_computation_type("nion.measure_shifts", MeasureShifts)
Symbolic.register_computation_type("nion.apply_shifts", ApplyShifts)
Symbolic.register_computation_type("nion.crop_multi_dimensional", Crop)

AxesChoice = Schema.entity("axis_choice", None, None, {})

for choice_id, choice_name in [('collection', 'Collection'), ('sequence', 'Sequence'), ('data', 'Data')]:
    axis_choice_entity = Schema.entity(choice_id, AxesChoice, None, {})
    DataStructure.DataStructure.register_entity(axis_choice_entity, entity_name=choice_name, entity_package_name=_("EELS Analysis"))
