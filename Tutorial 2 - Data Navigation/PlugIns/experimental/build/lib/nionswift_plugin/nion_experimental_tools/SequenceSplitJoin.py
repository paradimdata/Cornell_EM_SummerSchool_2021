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

_ = gettext.gettext


class SequenceJoin:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, src_list):
        try:
            self.__new_xdata = xd.sequence_join([data_item.xdata for data_item in src_list])
        except Exception as e:
            print(str(e))
            import traceback
            traceback.print_exc()

    def commit(self):
        self.computation.set_referenced_xdata("target", self.__new_xdata)


class SequenceSplit:
    def __init__(self, computation, **kwargs):
        self.computation = computation

    def execute(self, src):
        try:
            self.__new_xdata_list = xd.sequence_split(src.xdata)
        except Exception as e:
            print(str(e))
            import traceback
            traceback.print_exc()

    def commit(self):
        if self.__new_xdata_list:
            for i, xdata in enumerate(self.__new_xdata_list):
                self.computation.set_referenced_xdata(f"target_{i}", xdata)


class SequenceJoinMenuItem:
    menu_id = "_processing_menu"  # required, specify menu_id where this item will go
    menu_item_name = _("Join Sequence(s)")  # menu item name

    def __init__(self, api):
        self.__api = api

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        document_controller = window._document_controller
        selected_display_items = document_controller.selected_display_items
        data_items = list()

        # Check if it makes sense to copy display properties from the source to the result display item.
        # For line plots with multiple display layers we want to copy the display properties so that the joined item
        # look like the original display items. We copy the display properties of the first display item, but only
        # if the number of display layers is the same for all input display items.
        display_layers = None
        legend_position = None
        display_type = None
        copy_display_properties = False
        for i, display_item in enumerate(selected_display_items):
            data_item = display_item.data_items[0] if display_item and len(display_item.data_items) > 0 else None

            if data_item:
                data_items.append(data_item)
                if (len(display_item.data_items) == 1 and len(data_item.data_shape) > 1 and
                        (data_item.xdata.datum_dimension_count == 1 or display_item.display_type == 'line_plot')):
                    if i== 0:
                        display_layers = copy.deepcopy(display_item.display_layers)
                        legend_position = display_item.get_display_property('legend_position')
                        display_type = display_item.display_type
                        copy_display_properties = True
                    elif display_layers is not None:
                        copy_display_properties &= len(display_layers) == len(display_item.display_layers)

        if not data_items:
            return

        api_data_items = [Facade.DataItem(data_item) for data_item in data_items]

        result_data_item = self.__api.library.create_data_item_from_data(numpy.zeros((1, 1, 1)), title="Joined " + data_items[0].title)
        computation = self.__api.library.create_computation("nion.join_sequence",
                                                            inputs={"src_list": api_data_items},
                                                            outputs={"target": result_data_item})
        computation._computation.source = result_data_item._data_item
        result_display_item = document_controller.document_model.get_display_item_for_data_item(result_data_item._data_item)
        document_controller.show_display_item(result_display_item)

        if copy_display_properties:
            if display_layers is not None:
                result_display_item.display_layers = display_layers
            if legend_position is not None:
                result_display_item.set_display_property('legend_position', legend_position)
            if display_type is not None:
                result_display_item.display_type = display_type


class SequenceSplitMenuItem:
    menu_id = "_processing_menu"
    menu_item_name = _("Split Sequence")

    def __init__(self, api):
        self.__api = api

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        document_controller = window._document_controller
        display_item = document_controller.selected_display_item
        data_item = display_item.data_items[0] if display_item and len(display_item.data_items) > 0 else None

        if not data_item:
            return

        # Check if it makes sense to copy display properties from the source to the result display item.
        # For line plots with multiple display layers we want to copy the display properties so that the split items
        # look like the original display item. Exclude case where the display layers are generated from the sequence
        # dimension because in this case the display layers are not valid anymore.
        display_layers = None
        legend_position = None
        display_type = None
        if (len(display_item.data_items) == 1 and len(data_item.data_shape) > 2 and
                (data_item.xdata.datum_dimension_count == 1 or display_item.display_type == 'line_plot')):
            display_layers = copy.deepcopy(display_item.display_layers)
            legend_position = display_item.get_display_property('legend_position')
            display_type = display_item.display_type
        api_data_item = Facade.DataItem(data_item)

        if api_data_item.xdata.is_sequence:
            if api_data_item.xdata.data_shape[0] > 100:
                logging.error("Splitting sequences of more than 100 items is disabled for performance reasons.")
                return
            result_data_items = {f"target_{i}": self.__api.library.create_data_item(title=f"Split ({i}) of " + data_item.title) for i in range(api_data_item.xdata.data_shape[0])}
            computation = self.__api.library.create_computation("nion.split_sequence",
                                                                inputs={"src": api_data_item},
                                                                outputs=result_data_items)
            computation._computation.source = result_data_items["target_0"]._data_item

            for result_data_item in result_data_items.values():
                result_display_item = document_controller.document_model.get_display_item_for_data_item(result_data_item._data_item)
                document_controller.show_display_item(result_display_item)

                if display_layers is not None:
                    result_display_item.display_layers = display_layers
                if legend_position is not None:
                    result_display_item.set_display_property('legend_position', legend_position)
                if display_type is not None:
                    result_display_item.display_type = display_type


class SequenceSplitJoinExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.extension.sequence_split_join"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__join_menu_item_ref = api.create_menu_item(SequenceJoinMenuItem(api))
        self.__split_menu_item_ref = api.create_menu_item(SequenceSplitMenuItem(api))

    def close(self):
        self.__join_menu_item_ref.close()
        self.__split_menu_item_ref.close()

Symbolic.register_computation_type("nion.join_sequence", SequenceJoin)
Symbolic.register_computation_type("nion.split_sequence", SequenceSplit)
