# system imports
import gettext

import numpy as np

# local libraries
from nion.typeshed import API_1_0 as API
from nion.swift import Facade
from nion.swift.model import Graphics
from nion.swift.model import Symbolic

from .DataCache import DataCache

_ = gettext.gettext


class Map4DRGB:
    attributes = {"connection_type": "map"}

    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__api = computation.api
        if not hasattr(computation, 'data_cache'):
            def modify_data_fn(data):
                new_shape = data.shape[:2] + (-1,)
                return np.reshape(data, new_shape)
            computation.data_cache = DataCache(modify_data_fn=modify_data_fn)

        def create_panel_widget(ui, document_controller):
            def select_button_clicked(channel):
                graphics = document_controller.target_display.selected_graphics
                if not graphics:
                    return
                try:
                    while True:
                        self.computation._computation.remove_item_from_objects(f'map_regions_{channel}', 0)
                except IndexError:
                    pass
                for graphic in graphics:
                    self.computation._computation.insert_item_into_objects(f'map_regions_{channel}', 0, Symbolic.make_item(graphic._graphic))
                    title_suffix = f' ({channel.upper()})'
                    title = graphic.label or ''
                    if not title.endswith(title_suffix):
                        title += title_suffix
                        graphic.label = title

            def gamma_changed(channel, value):
                try:
                    value = float(value)
                except ValueError:
                    return
                gamma_variable = self.computation._computation._get_variable('gamma_' + channel)
                if gamma_variable.value != value:
                    gamma_variable.value = value

            def enabled_changed(channel, check_state):
                enabled_variable = self.computation._computation._get_variable('enabled_' + channel)
                enabled = check_state == 'checked'
                if enabled_variable.value != enabled:
                    enabled_variable.value = enabled

            def select_red_channel_button_clicked():
                select_button_clicked('r')

            def select_green_channel_button_clicked():
                select_button_clicked('g')

            def select_blue_channel_button_clicked():
                select_button_clicked('b')

            def gamma_red_finished(text):
                gamma_changed('r', text)

            def gamma_green_finished(text):
                gamma_changed('g', text)

            def gamma_blue_finished(text):
                gamma_changed('b', text)

            def enabled_red_changed(check_state):
                enabled_changed('r', check_state)

            def enabled_green_changed(check_state):
                enabled_changed('g', check_state)

            def enabled_blue_changed(check_state):
                enabled_changed('b', check_state)

            column = ui.create_column_widget()
            row0 = ui.create_row_widget()
            row1 = ui.create_row_widget()
            row2 = ui.create_row_widget()
            row3 = ui.create_row_widget()

            select_red_graphics_button = ui.create_push_button_widget('Red')
            select_green_graphics_button = ui.create_push_button_widget('Green')
            select_blue_graphics_button = ui.create_push_button_widget('Blue')

            gamma_red_field = ui.create_line_edit_widget()
            gamma_green_field = ui.create_line_edit_widget()
            gamma_blue_field = ui.create_line_edit_widget()

            gamma_red_field.text = self.computation._computation._get_variable('gamma_r').value
            gamma_green_field.text = self.computation._computation._get_variable('gamma_g').value
            gamma_blue_field.text = self.computation._computation._get_variable('gamma_b').value

            enabled_red = ui.create_check_box_widget('Enabled')
            enabled_green = ui.create_check_box_widget('Enabled')
            enabled_blue = ui.create_check_box_widget('Enabled')

            enabled_red.checked = self.computation._computation._get_variable('enabled_r').value
            enabled_green.checked = self.computation._computation._get_variable('enabled_g').value
            enabled_blue.checked = self.computation._computation._get_variable('enabled_b').value

            row0.add_spacing(10)
            row0.add(ui.create_label_widget('Select map graphic(s) for channel:'))
            row0.add_spacing(20)
            row0.add_stretch()
            row0.add(ui.create_label_widget('Gamma for channel:'))
            row0.add_spacing(10)

            row1.add_spacing(20)
            row1.add(select_red_graphics_button)
            row1.add_spacing(5)
            row1.add(enabled_red)
            row1.add_spacing(10)
            row1.add_stretch()
            row1.add(gamma_red_field)
            row1.add_spacing(10)

            row2.add_spacing(20)
            row2.add(select_green_graphics_button)
            row2.add_spacing(5)
            row2.add(enabled_green)
            row2.add_spacing(10)
            row2.add_stretch()
            row2.add(gamma_green_field)
            row2.add_spacing(10)

            row3.add_spacing(20)
            row3.add(select_blue_graphics_button)
            row3.add_spacing(5)
            row3.add(enabled_blue)
            row3.add_spacing(10)
            row3.add_stretch()
            row3.add(gamma_blue_field)
            row3.add_spacing(10)

            column.add_spacing(10)
            column.add(row0)
            column.add_spacing(3)
            column.add(row1)
            column.add_spacing(3)
            column.add(row2)
            column.add_spacing(3)
            column.add(row3)
            column.add_spacing(10)
            column.add_stretch()

            select_red_graphics_button.on_clicked = select_red_channel_button_clicked
            select_green_graphics_button.on_clicked = select_green_channel_button_clicked
            select_blue_graphics_button.on_clicked = select_blue_channel_button_clicked

            gamma_red_field.on_editing_finished = gamma_red_finished
            gamma_green_field.on_editing_finished = gamma_green_finished
            gamma_blue_field.on_editing_finished = gamma_blue_finished

            enabled_red.on_check_state_changed = enabled_red_changed
            enabled_green.on_check_state_changed = enabled_green_changed
            enabled_blue.on_check_state_changed = enabled_blue_changed

            return column

        self.computation._computation.create_panel_widget = create_panel_widget

    def convert_to_8_bit(self, data, gamma=1.0):
        data = data - np.amin(data)
        max_ = np.amax(data)
        data = data / (max_ if max_ != 0 else 1)
        data = 255*data**gamma
        return np.rint(data).astype(np.uint8)

    def execute(self, src, map_regions_r, map_regions_g, map_regions_b, gamma_r, gamma_g, gamma_b, enabled_r,
                enabled_g, enabled_b):
        try:
            src_data_item = src.data_item
            data = self.computation.data_cache.get_cached_data(src_data_item)
            rgb_data = np.zeros(src_data_item.xdata.data_shape[:2] + (3,), dtype=np.uint8)
            map_regions_rgb = (map_regions_b, map_regions_g, map_regions_r)
            channels_enabled = (enabled_b, enabled_g, enabled_r)
            gammas = (gamma_b, gamma_g, gamma_r)
            for i in range(len(map_regions_rgb)):
                if not channels_enabled[i]:
                    continue
                map_regions = map_regions_rgb[i]
                mask_data = np.zeros(src_data_item.xdata.data_shape[2:], dtype=np.bool)
                for region in map_regions:
                    mask_data = np.logical_or(mask_data, region.get_mask(src_data_item.xdata.data_shape[2:]))
                if mask_data.any():
                    ind = np.arange(mask_data.size)[mask_data.ravel()]
                    new_data = np.sum(data[..., ind], axis=(-1))
                    #y = np.unique(np.indices(mask_data.shape)[0][mask_data])
                    #x = np.unique(np.indices(mask_data.shape)[1][mask_data])
                    #new_data = np.sum(xdata.data[..., x][..., y, :], axis=(-2, -1))
                    rgb_data[..., i] = self.convert_to_8_bit(new_data, gammas[i])

                else:
                    rgb_data[..., i] = self.convert_to_8_bit(np.sum(data, axis=-1), gammas[i])

            self.__new_xdata = self.__api.create_data_and_metadata(rgb_data,
                                                                   dimensional_calibrations=src_data_item.dimensional_calibrations[:2],
                                                                   intensity_calibration=src_data_item.intensity_calibration)
            metadata = src_data_item.metadata.copy()
            metadata['nion.map_4d_rgb.parameters'] = {'src': src_data_item._data_item.write_to_dict(),
                                                      'map_regions_r': [region.write_to_dict() for region in map_regions_r],
                                                      'map_regions_g': [region.write_to_dict() for region in map_regions_g],
                                                      'map_regions_b': [region.write_to_dict() for region in map_regions_b],
                                                      'gamma_r': gamma_r,
                                                      'gamma_g': gamma_g,
                                                      'gamma_b': gamma_b,
                                                      'channels_enabled': list(channels_enabled)}
            metadata['nion.map_4d_rgb.parameters']['channels_enabled'].reverse()
            self.__new_xdata.metadata.update(metadata)
        except Exception as e:
            print(str(e))
            import traceback
            traceback.print_exc()

    def commit(self):
        self.computation.set_referenced_xdata('target', self.__new_xdata)


class Map4DRGBMenuItem:

    menu_id = "4d_tools_menu"  # required, specify menu_id where this item will go
    menu_name = _("4D Tools") # optional, specify default name if not a standard menu
    menu_before_id = "window_menu" # optional, specify before menu_id if not a standard menu
    menu_item_name = _("Map 4D RGB")  # menu item name

    def __init__(self, api):
        self.__api = api
        self.__computation_data_items = dict()
        self.__tool_tip_boxes = list()

    def __display_item_changed(self, display_item):
        data_item = display_item.data_item if display_item else None
        if data_item:
            tip_id = self.__computation_data_items.get(str(data_item.uuid))
            if tip_id:
                self.__show_tool_tips(tip_id)

    def __show_tool_tips(self, tip_id='source', timeout=30):
        for box in self.__tool_tip_boxes:
            box.remove_now()
        self.__tool_tip_boxes = list()
        if tip_id == 'source':
            text = ('Select one or multiple graphic(s) on the source data item per channel and click "Select" in the '
                    'computation panel (Window -> Computation).\nWithout a selected graphic, the whole frames will be '
                    'summed.')
        elif tip_id == 'map_4d':
            text = 'Move the "Pick" graphic to change the data slice in the source data item.'
        elif tip_id == 'wrong_shape':
            text = 'This computation only works for 4D-data.'
        else:
            return
        document_controller = self.__api.application.document_windows[0]
        workspace = document_controller._document_controller.workspace_controller
        box = workspace.pose_tool_tip_box(text, timeout)
        #box = document_controller.show_tool_tip_box(text, timeout)
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
            map_data_item = self.__api.library.create_data_item(title='Map 4D (RGB) of ' + data_item.title)
            # the following uses internal API and should not be used as example code.
            computation = document_controller.document_model.create_computation()
            computation.create_input_item("src", Symbolic.make_item(selected_display_item.get_display_data_channel_for_data_item(data_item)))
            computation.create_input_item("map_regions_r", Symbolic.make_item_list([]))
            computation.create_input_item("map_regions_g", Symbolic.make_item_list([]))
            computation.create_input_item("map_regions_b", Symbolic.make_item_list([]))
            computation.create_variable("gamma_r", value_type="real", value=1.0)
            computation.create_variable("gamma_g", value_type="real", value=1.0)
            computation.create_variable("gamma_b", value_type="real", value=1.0)
            computation.create_variable("enabled_r", value_type="boolean", value=True)
            computation.create_variable("enabled_g", value_type="boolean", value=True)
            computation.create_variable("enabled_b", value_type="boolean", value=True)
            computation.processing_id = "nion.map_4d_rgb.2"
            document_controller.document_model.set_data_item_computation(map_data_item._data_item, computation)
            map_display_item = document_controller.document_model.get_display_item_for_data_item(map_data_item._data_item)
            document_controller.show_display_item(map_display_item)
            graphic = Graphics.PointGraphic()
            graphic.label = "Pick"
            graphic.role = "collection_index"
            map_display_item.add_graphic(graphic)
            # see note above.
            self.__computation_data_items.update({str(data_item.uuid): 'source',
                                                  str(map_data_item._data_item.uuid): 'map_4d'})
            self.__show_tool_tips()
            self.__display_item_changed_event_listener = (
                           document_controller.focused_display_item_changed_event.listen(self.__display_item_changed))


class Map4DExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.extension.map_4d_rgb"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(Map4DRGBMenuItem(api))

    def close(self):
        self.__menu_item_ref.close()

Symbolic.register_computation_type('nion.map_4d_rgb.2', Map4DRGB)
