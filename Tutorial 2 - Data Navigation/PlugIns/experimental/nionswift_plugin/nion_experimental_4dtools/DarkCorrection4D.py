# system imports
import gettext
from nion.swift.model import Symbolic, DataItem
from nion.swift import Facade

# local libraries
from nion.typeshed import API_1_0 as API
from nion.data import xdata_1_0 as xd

import numpy as np
import uuid

from . import ImageChooser
from .DataCache import DataCache

_ = gettext.gettext


class TotalBin4D:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, src):
        try:
            self.__new_xdata = xd.sum(src.xdata, axis=(2, 3))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(str(e))
            raise

    def commit(self):
        self.computation.set_referenced_xdata('target', self.__new_xdata)


class DarkCorrection4D:
    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__api = computation.api
        if not hasattr(computation, 'data_cache'):
            computation.data_cache = DataCache()

        def create_panel_widget(ui, document_controller):
            def gain_mode_changed(current_item):
                if current_item != self.computation._computation._get_variable('gain_mode').value:
                    self.computation._computation._get_variable('gain_mode').value = current_item

            def bin_data_changed(check_state):
                if self.computation._computation._get_variable('bin_spectrum').value != (check_state == 'checked'):
                    self.computation._computation._get_variable('bin_spectrum').value = check_state == 'checked'

            def clear_gain_image():
                variable = self.computation._computation._get_variable('gain_image')
                if variable.objects_model.items:
                    variable.objects_model.remove_item(0)

            column = ui.create_column_widget()
            image_chooser, image_changed_listener = ImageChooser.make_image_chooser(document_controller._document_controller,
                                                                                    self.computation._computation,
                                                                                    self.computation._computation._get_variable('gain_image'))
            self.__image_changed_listener = image_changed_listener

            clear_gain_image_button = ui.create_push_button_widget('Clear')

            gain_mode_label = ui.create_label_widget('Gain correction mode: ')
            gain_mode_chooser = ui.create_combo_box_widget()
            gain_mode_chooser.items = ['auto', 'custom', 'off']
            bin_data_checkbox = ui.create_check_box_widget('Bin data to 1D')

            gain_mode_row = ui.create_row_widget()
            gain_mode_row.add(gain_mode_label)
            gain_mode_row.add(gain_mode_chooser)
            gain_mode_row.add_stretch()

            gain_mode_row2 = ui.create_row_widget()
            gain_mode_row2.add_stretch()
            gain_mode_row2._widget.add(image_chooser)
            gain_mode_row2.add_spacing(5)
            gain_mode_row2.add(clear_gain_image_button)

            bin_data_row = ui.create_row_widget()
            bin_data_row.add(bin_data_checkbox)
            bin_data_row.add_stretch()

            column.add(gain_mode_row)
            column.add_spacing(10)
            column.add(gain_mode_row2)
            column.add_spacing(10)
            column.add(bin_data_row)
            column.add_stretch()

            gain_mode_chooser.current_item = self.computation._computation._get_variable('gain_mode').value
            bin_data_checkbox.checked = self.computation._computation._get_variable('bin_spectrum').value

            clear_gain_image_button.on_clicked = clear_gain_image
            gain_mode_chooser.on_current_item_changed = gain_mode_changed
            bin_data_checkbox.on_check_state_changed = bin_data_changed

            return column

        self.computation._computation.create_panel_widget = create_panel_widget

    def execute(self, src1, src2, dark_area_region, crop_region, bin_spectrum, gain_image, gain_mode):
        try:
            xdata = src1.xdata
            metadata = src1.metadata.copy()
            data_shape = np.array(xdata.data.shape)
            dark_area = np.rint(np.array(dark_area_region.bounds) * np.array((data_shape[:2], data_shape[:2]))).astype(np.int)
            crop_area = np.rint(np.array(crop_region.bounds) * np.array((data_shape[2:], data_shape[2:]))).astype(np.int)

            dark_image = xd.sum(xdata[dark_area[0, 0]:dark_area[0, 0]+dark_area[1, 0],
                                      dark_area[0, 1]:dark_area[0, 1]+dark_area[1, 1],
                                      crop_area[0, 0]:crop_area[0, 0]+crop_area[1, 0],
                                      crop_area[0, 1]:crop_area[0, 1]+crop_area[1, 1]], axis=(0, 1))/(dark_area[1,0]*dark_area[1,1])

            self.__new_xdata = xdata[..., crop_area[0, 0]:crop_area[0, 0]+crop_area[1, 0],
                                          crop_area[0, 1]:crop_area[0, 1]+crop_area[1, 1]] - dark_image

            current_gain_image_uuid = metadata.get('hardware_source', {}).get('current_gain_image')
            current_gain_image = None
            if current_gain_image_uuid:
                current_gain_image = self.__api.library.get_data_item_by_uuid(uuid.UUID(current_gain_image_uuid))
            if metadata.get('hardware_source', {}).get('is_gain_corrected'):
                if gain_mode in ('custom', 'off') and current_gain_image:
                    if current_gain_image.xdata.data_shape == xdata.data_shape[2:]:
                        self.__new_xdata /= current_gain_image.xdata[crop_area[0, 0]:crop_area[0, 0]+crop_area[1, 0],
                                                                     crop_area[0, 1]:crop_area[0, 1]+crop_area[1, 1]]

            if ((gain_mode == 'auto' and not metadata.get('hardware_source', {}).get('is_gain_corrected') and current_gain_image) or
                (gain_mode == 'custom' and gain_image)):

                gain_xdata = gain_image[0].xdata if gain_mode == 'custom' else current_gain_image.xdata

                if gain_xdata.data_shape == self.__new_xdata.data_shape[2:]:
                    self.__new_xdata *= gain_xdata
                elif gain_xdata.data_shape == xdata.data_shape[2:]:
                    self.__new_xdata *= gain_xdata[crop_area[0, 0]:crop_area[0, 0]+crop_area[1, 0],
                                                   crop_area[0, 1]:crop_area[0, 1]+crop_area[1, 1]]
                else:
                    raise ValueError('Shape of gain image has to match last two dimensions of input data.')
                del gain_xdata

            if bin_spectrum:
                self.__new_xdata = xd.sum(self.__new_xdata, axis=2)

            metadata['nion.dark_correction_4d.parameters'] = {'src1': src1._data_item.write_to_dict(),
                                                              'src2': src2._data_item.write_to_dict(),
                                                              'dark_area_region': dark_area_region._graphic.write_to_dict(),
                                                              'crop_region': crop_region._graphic.write_to_dict(),
                                                              'bottom_dark_region_region': crop_region._graphic.write_to_dict(),
                                                              'bin_spectrum': bin_spectrum,
                                                              'gain_image': gain_image[0].data_item.write_to_dict() if gain_image else None,
                                                              'gain_mode': gain_mode}
            self.__new_xdata.metadata.update(metadata)
        except Exception as e:
            print(str(e))
            import traceback
            traceback.print_exc()

    def commit(self):
        self.computation.set_referenced_xdata('target', self.__new_xdata)

class DarkCorrection4DMenuItem:

    menu_id = "4d_tools_menu"  # required, specify menu_id where this item will go
    menu_name = _("4D Tools") # optional, specify default name if not a standard menu
    menu_before_id = "window_menu" # optional, specify before menu_id if not a standard menu
    menu_item_name = _("4D Dark Correction")  # menu item name

    #DocumentModel.DocumentModel.register_processing_descriptions(correct_dark_processing_descriptions)
    #DocumentModel.DocumentModel.register_processing_descriptions(calculate_average_processing_descriptions)
    def __init__(self, api):
        self.__api = api
        self.__computation_data_items = dict()
        self.__tool_tip_boxes = list()

    def __display_item_changed(self, display_item):
        data_item = display_item.data_item if display_item else None
        if data_item:
            tip_id = self.__computation_data_items.get(data_item)
            if tip_id:
                self.__show_tool_tips(tip_id)

    def __show_tool_tips(self, tip_id='source', timeout=30):
        for box in self.__tool_tip_boxes:
            box.remove_now()
        self.__tool_tip_boxes = list()
        if tip_id == 'source':
            text = 'Use the "Crop" graphic to crop the camera images.\nSelect the "Total bin..." or "4D dark correction..." data item for further options.'
        elif tip_id == 'total bin':
            text = 'Use the "Dark subtract area" graphic to select the are that was not illuminated. The sum of all camera frames will be the dark image.\nSelect the source data item or the "4D dark correction..." data item for further options.'
        elif tip_id == 'corrected':
            text = 'In the "Computation" panel (Window -> Computation) you find the settings. Custom gain correction mode uses the gain image from the drag-and-drop area.\n"Bin data to 1D" will ouput 1D spectra instead of 2D camera frames in the result data item.\nSelect the source data item or the "Total bin..." data item for further options.'
        elif tip_id == 'wrong_shape':
            text = 'This computation only works for 4D-data.'
        else:
            return
        document_controller = self.__api.application.document_windows[0]
        #box = document_controller.show_tool_tip_box(text, timeout)
        workspace = document_controller._document_controller.workspace_controller
        box = workspace.pose_tool_tip_box(text, timeout)
        self.__tool_tip_boxes.append(box)

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        document_controller = window._document_controller
        selected_display_item = document_controller.selected_display_item
        data_item = (selected_display_item.data_items[0] if
                     selected_display_item and len(selected_display_item.data_items) > 0 else None)

        if data_item:
            api_data_item = Facade.DataItem(data_item)
            if not api_data_item.xdata.is_data_4d:
                self.__show_tool_tips('wrong_shape')
                return
            total_bin_data_item = self.__api.library.create_data_item(title='Total bin 4D of ' + data_item.title)
            computation = self.__api.library.create_computation('nion.total_bin_4d_SI',
                                                                inputs={'src': api_data_item},
                                                                outputs={'target': total_bin_data_item})
            computation._computation.source = total_bin_data_item._data_item
            #computation._computation.mark_update()

            total_bin_display_item = document_controller.document_model.get_display_item_for_data_item(
                                                                                                  total_bin_data_item._data_item)
            document_controller.show_display_item(total_bin_display_item)
            dark_subtract_area_graphic = total_bin_data_item.add_rectangle_region(0.8, 0.5, 0.4, 1.0)
            dark_subtract_area_graphic.label = 'Dark subtract area'
            crop_region = api_data_item.add_rectangle_region(0.5, 0.5, 1.0, 1.0)
            crop_region.label = 'Crop'

            dark_subtract_area_graphic._graphic.is_bounds_constrained = True
            crop_region._graphic.is_bounds_constrained = True
            dark_corrected_data_item = Facade.DataItem(DataItem.DataItem(large_format=True))
            self.__api.library._document_model.append_data_item(dark_corrected_data_item._data_item)
            dark_corrected_data_item._data_item.session_id = self.__api.library._document_model.session_id
            dark_corrected_data_item.title = '4D dark correction of ' + data_item.title
            computation = self.__api.library.create_computation('nion.dark_correction_4d',
                                                                inputs={'src1': api_data_item,
                                                                        'src2': total_bin_data_item,
                                                                        'dark_area_region': dark_subtract_area_graphic,
                                                                        'crop_region': crop_region,
                                                                        'bin_spectrum': True,
                                                                        'gain_image': [],
                                                                        'gain_mode': 'custom'},
                                                                outputs={'target': dark_corrected_data_item})
            computation._computation.source = dark_corrected_data_item._data_item
            dark_corrected_display_item = document_controller.document_model.get_display_item_for_data_item(
                                                                                             dark_corrected_data_item._data_item)
            document_controller.show_display_item(dark_corrected_display_item)
            self.__computation_data_items.update({data_item: 'source',
                                                  total_bin_data_item._data_item: 'total bin',
                                                  dark_corrected_data_item._data_item: 'corrected'})
            self.__show_tool_tips()
            self.__display_item_changed_event_listener = (
                           document_controller.focused_display_item_changed_event.listen(self.__display_item_changed))

class DarkCorrection4DExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.extension.4d_dark_correction"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(DarkCorrection4DMenuItem(api))

    def close(self):
        self.__menu_item_ref.close()

Symbolic.register_computation_type('nion.total_bin_4d_SI', TotalBin4D)
Symbolic.register_computation_type('nion.dark_correction_4d', DarkCorrection4D)
