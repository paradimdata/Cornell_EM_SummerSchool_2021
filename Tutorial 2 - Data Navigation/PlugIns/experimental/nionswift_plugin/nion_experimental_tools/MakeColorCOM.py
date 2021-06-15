import gettext
import logging

import numpy
import scipy.optimize

from nion.data import DataAndMetadata
from nion.swift.model import Symbolic
from nion.typeshed import API_1_0 as API
from nion.swift import Facade

_ = gettext.gettext


class MakeColorCOM:
    label = _("Make color COM image")
    inputs = {"src": {"label": _("COM Data Item")},
              "com_x_index": {"label": _("COM x-slice")},
              "com_y_index": {"label": _("COM y-slice")},
              "magnitude_min": {"label": _("Magnitude min (percentile)")},
              "magnitude_max": {"label": _("Magnitude max (percentile)")},
              "rotation": {"label": _("Rotation")},
              }
    outputs = {"output": {"label": _("Color COM")}}

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__result_xdata = None

    def __calculate_curl(self, rotation, com_x, com_y):
        com_x_rotated = com_x * numpy.cos(rotation) - com_y * numpy.sin(rotation)
        com_y_rotated = com_x * numpy.sin(rotation) + com_y * numpy.cos(rotation)
        curl_com = numpy.gradient(com_y_rotated, axis=1) - numpy.gradient(com_x_rotated, axis=0)
        return numpy.mean(curl_com**2)

    def execute(self, src, com_x_index, com_y_index, magnitude_min, magnitude_max, rotation, **kwargs) -> None:
        try:
            com_xdata = src.xdata
            assert com_xdata.is_datum_2d
            assert com_xdata.is_sequence or com_xdata.is_collection
            com_x = com_xdata.data[com_x_index]
            com_y = com_xdata.data[com_y_index]

            if not rotation or rotation == "None":
                res = scipy.optimize.minimize_scalar(self.__calculate_curl, 0, args=(com_x, com_y), bounds=(0, numpy.pi*2), method='bounded')
                if res.success:
                    rotation = res.x
                    logging.info(f'Calculated optimal roation: {rotation*180/numpy.pi:.1f} degree.')
                else:
                    logging.warning(f'Could not find the optimal rotation. Optimize error: {res.message}\nUsing rotation=0 as default.')
                    rotation = 0
            else:
                rotation = float(rotation) / 180.0 * numpy.pi

            com_x_rotated = com_x * numpy.cos(rotation) - com_y * numpy.sin(rotation)
            com_y_rotated = com_x * numpy.sin(rotation) + com_y * numpy.cos(rotation)

            com_magnitude = numpy.sqrt(com_x_rotated**2 + com_y_rotated**2)
            com_angle = numpy.arctan2(com_y_rotated, com_x_rotated)

            com_angle += numpy.pi
            com_angle *= 255.0 * 0.5 / numpy.pi
            com_angle = numpy.rint(com_angle).astype(int)

            if magnitude_min != 0 or magnitude_max != 100:
                percentile_min, percentile_max = numpy.percentile(com_magnitude, (magnitude_min, magnitude_max))
            if magnitude_min != 0:
                com_magnitude[com_magnitude < percentile_min] = percentile_min
            if magnitude_max != 100:
                com_magnitude[com_magnitude > percentile_max] = percentile_max

            com_magnitude -= numpy.amin(com_magnitude)
            com_magnitude *= 1.0 / numpy.amax(com_magnitude)
            com_magnitude = com_magnitude[..., numpy.newaxis]
            com_magnitude = numpy.repeat(com_magnitude, 3, axis=-1)
            rgb_angle = hsv_cyclic_colormap[com_angle]
            combined = ((com_magnitude * rgb_angle)).astype(numpy.uint8)

            self.__result_xdata = DataAndMetadata.new_data_and_metadata(combined,
                                                                        dimensional_calibrations=com_xdata.dimensional_calibrations[1:])
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        self.computation.set_referenced_xdata("output", self.__result_xdata)


class MakeColorCOMMenuItem:
    menu_id = "_processing_menu"
    menu_item_name = _("Make color COM image")

    def __init__(self, api):
        self.__api = api

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        document_controller = window._document_controller
        display_item = document_controller.selected_display_item
        data_item = display_item.data_items[0] if display_item and len(display_item.data_items) > 0 else None

        if not data_item:
            return

        api_data_item = Facade.DataItem(data_item)

        if api_data_item.xdata.is_sequence and api_data_item.xdata.datum_dimension_count == 2:
            result_data_item = {"output": self.__api.library.create_data_item(title="Color COM image of " + data_item.title)}
            self.__api.library.create_computation("nion.make_color_com",
                                                  inputs={"src": api_data_item,
                                                          "com_x_index": 0,
                                                          "com_y_index": 1,
                                                          "magnitude_min": 0,
                                                          "magnitude_max": 100,
                                                          "rotation": "None"},
                                                  outputs=result_data_item)


class MakeColorCOMExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.extension.make_color_com"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__idpc_menu_item_ref = api.create_menu_item(MakeColorCOMMenuItem(api))

    def close(self):
        self.__idpc_menu_item_ref.close()


Symbolic.register_computation_type("nion.make_color_com", MakeColorCOM)

hsv_cyclic_colormap = numpy.array( [[255, 0, 0],
                                    [255, 6, 0],
                                    [255, 12, 0],
                                    [255, 18, 0],
                                    [255, 24, 0],
                                    [255, 30, 0],
                                    [255, 35, 0],
                                    [255, 41, 0],
                                    [255, 47, 0],
                                    [255, 53, 0],
                                    [255, 59, 0],
                                    [255, 65, 0],
                                    [255, 71, 0],
                                    [255, 77, 0],
                                    [255, 83, 0],
                                    [255, 89, 0],
                                    [255, 95, 0],
                                    [255, 100, 0],
                                    [255, 106, 0],
                                    [255, 112, 0],
                                    [255, 118, 0],
                                    [255, 124, 0],
                                    [255, 130, 0],
                                    [255, 136, 0],
                                    [255, 142, 0],
                                    [255, 148, 0],
                                    [255, 154, 0],
                                    [255, 159, 0],
                                    [255, 165, 0],
                                    [255, 171, 0],
                                    [255, 177, 0],
                                    [255, 183, 0],
                                    [255, 189, 0],
                                    [255, 195, 0],
                                    [255, 201, 0],
                                    [255, 207, 0],
                                    [255, 213, 0],
                                    [255, 219, 0],
                                    [255, 224, 0],
                                    [255, 230, 0],
                                    [255, 236, 0],
                                    [254, 241, 0],
                                    [252, 245, 0],
                                    [250, 249, 0],
                                    [248, 253, 0],
                                    [244, 255, 0],
                                    [238, 255, 0],
                                    [232, 255, 0],
                                    [226, 255, 0],
                                    [221, 255, 0],
                                    [215, 255, 0],
                                    [209, 255, 0],
                                    [203, 255, 0],
                                    [197, 255, 0],
                                    [191, 255, 0],
                                    [185, 255, 0],
                                    [179, 255, 0],
                                    [173, 255, 0],
                                    [167, 255, 0],
                                    [162, 255, 0],
                                    [156, 255, 0],
                                    [150, 255, 0],
                                    [144, 255, 0],
                                    [138, 255, 0],
                                    [132, 255, 0],
                                    [126, 255, 0],
                                    [120, 255, 0],
                                    [114, 255, 0],
                                    [108, 255, 0],
                                    [102, 255, 0],
                                    [97, 255, 0],
                                    [91, 255, 0],
                                    [85, 255, 0],
                                    [79, 255, 0],
                                    [73, 255, 0],
                                    [67, 255, 0],
                                    [61, 255, 0],
                                    [55, 255, 0],
                                    [49, 255, 0],
                                    [43, 255, 0],
                                    [37, 255, 0],
                                    [32, 255, 0],
                                    [26, 255, 0],
                                    [20, 255, 0],
                                    [14, 255, 0],
                                    [8, 255, 0],
                                    [6, 255, 4],
                                    [4, 255, 8],
                                    [2, 255, 12],
                                    [0, 255, 16],
                                    [0, 255, 22],
                                    [0, 255, 27],
                                    [0, 255, 33],
                                    [0, 255, 39],
                                    [0, 255, 45],
                                    [0, 255, 51],
                                    [0, 255, 57],
                                    [0, 255, 63],
                                    [0, 255, 69],
                                    [0, 255, 75],
                                    [0, 255, 81],
                                    [0, 255, 87],
                                    [0, 255, 92],
                                    [0, 255, 98],
                                    [0, 255, 104],
                                    [0, 255, 110],
                                    [0, 255, 116],
                                    [0, 255, 122],
                                    [0, 255, 128],
                                    [0, 255, 134],
                                    [0, 255, 140],
                                    [0, 255, 146],
                                    [0, 255, 151],
                                    [0, 255, 157],
                                    [0, 255, 163],
                                    [0, 255, 169],
                                    [0, 255, 175],
                                    [0, 255, 181],
                                    [0, 255, 187],
                                    [0, 255, 193],
                                    [0, 255, 199],
                                    [0, 255, 205],
                                    [0, 255, 211],
                                    [0, 255, 216],
                                    [0, 255, 222],
                                    [0, 255, 228],
                                    [0, 255, 234],
                                    [0, 255, 240],
                                    [0, 255, 246],
                                    [0, 255, 252],
                                    [0, 252, 255],
                                    [0, 246, 255],
                                    [0, 240, 255],
                                    [0, 234, 255],
                                    [0, 229, 255],
                                    [0, 223, 255],
                                    [0, 217, 255],
                                    [0, 211, 255],
                                    [0, 205, 255],
                                    [0, 199, 255],
                                    [0, 193, 255],
                                    [0, 187, 255],
                                    [0, 181, 255],
                                    [0, 175, 255],
                                    [0, 170, 255],
                                    [0, 164, 255],
                                    [0, 158, 255],
                                    [0, 152, 255],
                                    [0, 146, 255],
                                    [0, 140, 255],
                                    [0, 134, 255],
                                    [0, 128, 255],
                                    [0, 122, 255],
                                    [0, 116, 255],
                                    [0, 110, 255],
                                    [0, 105, 255],
                                    [0, 99, 255],
                                    [0, 93, 255],
                                    [0, 87, 255],
                                    [0, 81, 255],
                                    [0, 75, 255],
                                    [0, 69, 255],
                                    [0, 63, 255],
                                    [0, 57, 255],
                                    [0, 51, 255],
                                    [0, 45, 255],
                                    [0, 40, 255],
                                    [0, 34, 255],
                                    [0, 28, 255],
                                    [0, 22, 255],
                                    [0, 16, 255],
                                    [2, 12, 255],
                                    [4, 8, 255],
                                    [6, 4, 255],
                                    [8, 0, 255],
                                    [14, 0, 255],
                                    [19, 0, 255],
                                    [25, 0, 255],
                                    [31, 0, 255],
                                    [37, 0, 255],
                                    [43, 0, 255],
                                    [49, 0, 255],
                                    [55, 0, 255],
                                    [61, 0, 255],
                                    [67, 0, 255],
                                    [73, 0, 255],
                                    [79, 0, 255],
                                    [84, 0, 255],
                                    [90, 0, 255],
                                    [96, 0, 255],
                                    [102, 0, 255],
                                    [108, 0, 255],
                                    [114, 0, 255],
                                    [120, 0, 255],
                                    [126, 0, 255],
                                    [132, 0, 255],
                                    [138, 0, 255],
                                    [144, 0, 255],
                                    [149, 0, 255],
                                    [155, 0, 255],
                                    [161, 0, 255],
                                    [167, 0, 255],
                                    [173, 0, 255],
                                    [179, 0, 255],
                                    [185, 0, 255],
                                    [191, 0, 255],
                                    [197, 0, 255],
                                    [203, 0, 255],
                                    [208, 0, 255],
                                    [214, 0, 255],
                                    [220, 0, 255],
                                    [226, 0, 255],
                                    [232, 0, 255],
                                    [238, 0, 255],
                                    [244, 0, 255],
                                    [248, 0, 253],
                                    [250, 0, 249],
                                    [252, 0, 245],
                                    [254, 0, 241],
                                    [255, 0, 237],
                                    [255, 0, 231],
                                    [255, 0, 225],
                                    [255, 0, 219],
                                    [255, 0, 213],
                                    [255, 0, 207],
                                    [255, 0, 201],
                                    [255, 0, 195],
                                    [255, 0, 189],
                                    [255, 0, 183],
                                    [255, 0, 177],
                                    [255, 0, 172],
                                    [255, 0, 166],
                                    [255, 0, 160],
                                    [255, 0, 154],
                                    [255, 0, 148],
                                    [255, 0, 142],
                                    [255, 0, 136],
                                    [255, 0, 130],
                                    [255, 0, 124],
                                    [255, 0, 118],
                                    [255, 0, 113],
                                    [255, 0, 107],
                                    [255, 0, 101],
                                    [255, 0, 95],
                                    [255, 0, 89],
                                    [255, 0, 83],
                                    [255, 0, 77],
                                    [255, 0, 71],
                                    [255, 0, 65],
                                    [255, 0, 59],
                                    [255, 0, 53],
                                    [255, 0, 48],
                                    [255, 0, 42],
                                    [255, 0, 36],
                                    [255, 0, 30],
                                    [255, 0, 24]], dtype=int)