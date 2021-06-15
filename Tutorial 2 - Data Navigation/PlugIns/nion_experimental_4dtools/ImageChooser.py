# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:50:41 2018

@author: Andreas
"""
from nion.utils import Binding
from nion.swift import MimeTypes
from nion.swift import DataItemThumbnailWidget
from nion.utils import Geometry
import uuid
import copy

def make_image_chooser(document_controller, computation, variable):
    ui = document_controller.ui
    document_model = document_controller.document_model
    column = ui.create_column_widget()
    row = ui.create_row_widget()
    label_column = ui.create_column_widget()
    label_widget = ui.create_label_widget(variable.display_label, properties={"width": 80})
    label_widget.bind_text(Binding.PropertyBinding(variable, "display_label"))
    label_column.add(label_widget)
    label_column.add_stretch()
    row.add(label_column)
    row.add_spacing(8)

    def drop_mime_data(mime_data, x, y):
        if mime_data.has_format(MimeTypes.DISPLAY_ITEM_MIME_TYPE):
            display_item_uuid = uuid.UUID(mime_data.data_as_string(MimeTypes.DISPLAY_ITEM_MIME_TYPE))
            display_item = document_model.get_display_item_by_uuid(display_item_uuid)
            data_item = display_item.data_item if display_item else None
            if data_item:
                variable_specifier = document_model.get_object_specifier(display_item.display_data_channel)
                if variable.objects_model.items:
                    variable.objects_model.remove_item(0)
                variable.objects_model.append_item(variable_specifier)
                return "copy"
        return None

    def data_item_delete():
        if variable.objects_model.items:
            variable.objects_model.remove_item(0)

    display_item = None
    if variable.object_specifiers:
        base_variable_specifier = copy.copy(variable.object_specifiers[0])
        bound_data_source = document_model.resolve_object_specifier(base_variable_specifier)
        data_item = bound_data_source.value.data_item if bound_data_source else None
        display_item = document_model.get_display_item_for_data_item(data_item)
    data_item_thumbnail_source = DataItemThumbnailWidget.DataItemThumbnailSource(ui, display_item=display_item)
    data_item_chooser_widget = DataItemThumbnailWidget.ThumbnailWidget(ui, data_item_thumbnail_source, Geometry.IntSize(80, 80))

    def thumbnail_widget_drag(mime_data, thumbnail, hot_spot_x, hot_spot_y):
        # use this convoluted base object for drag so that it doesn't disappear after the drag.
        try:
            column.drag(mime_data, thumbnail, hot_spot_x, hot_spot_y)
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()

    data_item_chooser_widget.on_drag = thumbnail_widget_drag
    data_item_chooser_widget.on_drop_mime_data = drop_mime_data
    data_item_chooser_widget.on_delete = data_item_delete

    def property_changed(key):
        if key == "object_specifiers":
            if variable.object_specifiers:
                base_variable_specifier = copy.copy(variable.object_specifiers[0])
                bound_data_item = document_model.resolve_object_specifier(base_variable_specifier)
                data_item = bound_data_item.value.data_item if bound_data_item else None
                display_item = document_model.get_display_item_for_data_item(data_item)
                data_item_thumbnail_source.set_display_item(display_item)
            else:
                data_item_thumbnail_source.set_display_item(None)

    property_changed_listener = variable.property_changed_event.listen(property_changed)
    row.add(data_item_chooser_widget)
    row.add_stretch()
    column.add(row)
    column.add_spacing(4)
    return column, [property_changed_listener]
