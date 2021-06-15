# system imports
import gettext

# local libraries
from nion.typeshed import API_1_0 as API

_ = gettext.gettext


class AlignToCenterMenuItem:

    menu_id = "_edit_menu"  # required, specify menu_id where this item will go
    menu_item_name = _("Align Graphic to Center")  # menu item name

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        target_display = window.target_display
        if target_display:
            _display_item = target_display._display_item
            if _display_item.graphic_selection.has_selection:
                graphics = [_display_item.graphics[index] for index in _display_item.graphic_selection.indexes]
                for graphic in graphics:
                    if hasattr(graphic, "position"):
                        graphic.position = (0.5, 0.5)
                    if hasattr(graphic, "center"):
                        graphic.center = (0.5, 0.5)


class GraphicsToolsExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nionswift.graphics_tools"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(AlignToCenterMenuItem())

    def close(self):
        self.__menu_item_ref.close()
