import gettext

import numpy

from nion.data import DataAndMetadata
from nion.swift.model import Symbolic
from nion.typeshed import API_1_0 as API
from nion.swift import Facade

_ = gettext.gettext


class MakeIDPC:
    label = _("Make iDPC from DPC")
    inputs = {"src": {"label": _("DPC Data Item")},
              "gradient_x_index": {"label": _("DPC x-slice")},
              "gradient_y_index": {"label": _("DPC y-slice")},
              }
    outputs = {"output": {"label": _("iDPC")}}

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__result_xdata = None

    def execute(self, src, gradient_x_index, gradient_y_index, **kwargs) -> None:
        try:
            dpc_xdata = src.xdata
            assert dpc_xdata.is_datum_2d
            assert dpc_xdata.is_sequence
            gradx = dpc_xdata.data[gradient_x_index]
            grady = dpc_xdata.data[gradient_y_index]
            freq_v = numpy.fft.fftfreq(gradx.shape[-2], d=dpc_xdata.dimensional_calibrations[-2].scale)
            freq_u = numpy.fft.fftfreq(gradx.shape[-1], d=dpc_xdata.dimensional_calibrations[-1].scale)
            freqs = numpy.meshgrid(freq_u, freq_v)

            fft_idpc = (numpy.fft.fft2(gradx) * freqs[0] + numpy.fft.fft2(grady) * freqs[1]) / (1j * (freqs[0]**2 + freqs[1]**2))
            fft_idpc[numpy.isnan(fft_idpc)] = 0
            self.__result_xdata = DataAndMetadata.new_data_and_metadata(numpy.real(numpy.fft.ifft2(fft_idpc)),
                                                                        intensity_calibration=dpc_xdata.intensity_calibration,
                                                                        dimensional_calibrations=dpc_xdata.dimensional_calibrations[1:])
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            raise

    def commit(self):
        self.computation.set_referenced_xdata("output", self.__result_xdata)


class MakeIDPCMenuItem:
    menu_id = "_processing_menu"
    menu_item_name = _("Make iDPC from DPC")

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
            result_data_item = {"output": self.__api.library.create_data_item(title="iDPC of " + data_item.title)}
            self.__api.library.create_computation("nion.make_idpc",
                                                  inputs={"src": api_data_item,
                                                          "gradient_x_index": 0,
                                                          "gradient_y_index": 1},
                                                  outputs=result_data_item)


class MakeIDPCExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.extension.make_idpc"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__idpc_menu_item_ref = api.create_menu_item(MakeIDPCMenuItem(api))

    def close(self):
        self.__idpc_menu_item_ref.close()


Symbolic.register_computation_type("nion.make_idpc", MakeIDPC)