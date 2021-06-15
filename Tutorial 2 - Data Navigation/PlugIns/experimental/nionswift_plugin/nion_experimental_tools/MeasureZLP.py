# system imports
import gettext

# third part imports
import numpy

# local libraries
# None

_ = gettext.gettext


import math
import numpy
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.stats


def estimate_zlp_amplitude_position_width_fit_spline(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    assert len(d.shape) == 1 and d.shape[0] > 1
    gaussian = lambda x, a, b, c: a*numpy.exp(-(x-b)**2/(2*c**2))
    d_max = numpy.amax(d)
    # first fit a bspline to the data
    s = scipy.interpolate.splrep(range(d.shape[-1]), d - d_max / 2)
    # assuming bspline has two roots, use them to estimate FWHM
    r = scipy.interpolate.sproot(s)
    if len(r) == 2:
        fwhm = r[1] - r[0]
        c = fwhm / (2 * math.sqrt(2 * math.log(2)))
        # now fit the gaussian to the data, using the amplitude, std dev, and bspline position as estimates (10%)
        popt, pcov = scipy.optimize.curve_fit(gaussian, range(d.shape[0]), d, bounds=([d_max * 0.9, r[0], c * 0.9], [d_max * 1.1, r[1], c * 1.1]))
        return popt
    return numpy.nan, numpy.nan, numpy.nan


def estimate_zlp_amplitude_position_width_counting(d):
    # estimate the ZLP, assumes the peak value is the ZLP and that the ZLP is the only gaussian feature in the data
    assert len(d.shape) == 1 and d.shape[0] > 1
    mx_pos = numpy.argmax(d)
    mx = d[mx_pos]
    half_mx = mx/2
    left_pos = mx_pos - sum(d[:mx_pos] > half_mx)
    right_pos = mx_pos + sum(d[mx_pos:] > half_mx)
    return mx, mx_pos, left_pos, right_pos


def measure_zlp_fit_spline(api, window):
    """Attaches the measure ZLP computation to the target data item in the window."""
    target_data_item = window.target_data_item
    if target_data_item and target_data_item.display_xdata.is_data_1d:
        for graphic in target_data_item.graphics:
            if graphic.graphic_type == "interval-graphic" and graphic.graphic_id == "zlp_interval":
                target_data_item.remove_region(graphic)
                break
        data = target_data_item.display_xdata.data
        amplitude, pos, width = estimate_zlp_amplitude_position_width_fit_spline(data)
        if numpy.isfinite(amplitude) and numpy.isfinite(width) and numpy.isfinite(pos):
            fwhm = width * (2 * math.sqrt(2 * math.log(2)))
            start = (pos - fwhm/2) / data.shape[-1]
            end = (pos + fwhm/2) / data.shape[-1]
            zlp_interval = target_data_item.add_interval_region(start, end)
            zlp_interval.interval = start, end
            zlp_interval.graphic_id = "zlp_interval"


def measure_zlp_count_pixels(api, window):
    """Attaches the measure ZLP computation to the target data item in the window."""
    target_data_item = window.target_data_item
    if target_data_item and target_data_item.display_xdata.is_data_1d:
        for graphic in target_data_item.graphics:
            if graphic.graphic_type == "interval-graphic" and graphic.graphic_id == "zlp_interval_2":
                target_data_item.remove_region(graphic)
                break
        data = target_data_item.display_xdata.data
        amplitude, pos, left, right = estimate_zlp_amplitude_position_width_counting(data)
        start = left / data.shape[-1]
        end = right / data.shape[-1]
        zlp_interval = target_data_item.add_interval_region(start, end)
        zlp_interval.interval = start, end
        zlp_interval.graphic_id = "zlp_interval_2"
        zlp_interval._graphic.color = "#0F0"


class MeasureZLPFitSplineMenuItemDelegate:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Measure ZLP (Fit Spline)")  # menu item name

    def menu_item_execute(self, window):
        measure_zlp_fit_spline(self.__api, window)


class MeasureZLPCountPixelsMenuItemDelegate:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Measure ZLP (Count Pixels)")  # menu item name

    def menu_item_execute(self, window):
        measure_zlp_count_pixels(self.__api, window)


class Mark0eVMenuItemDelegate:

    def __init__(self, api):
        self.__api = api
        self.menu_id = "eels_menu"  # required, specify menu_id where this item will go
        self.menu_name = _("EELS")  # optional, specify default name if not a standard menu
        self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
        self.menu_item_name = _("Mark 0eV")  # menu item name

    def menu_item_execute(self, window):
        target_data_item = window.target_data_item
        if target_data_item and target_data_item.display_xdata.is_data_1d:
            for graphic in target_data_item.graphics:
                if graphic.graphic_type == "interval-graphic" and graphic.graphic_id == "channel_0eV":
                    target_data_item.remove_region(graphic)
                    break
            calibration = target_data_item.display_xdata.dimensional_calibrations[-1]
            channel = calibration.convert_from_calibrated_value(0) / target_data_item.display_xdata.dimensional_shape[-1]
            channel_graphic = target_data_item.add_interval_region(channel, channel)
            channel_graphic.graphic_id = "channel_0eV"
            channel_graphic._graphic.color = "#F00"
            channel_graphic._graphic.is_position_locked = True
            channel_graphic._graphic.is_shape_locked = True
            channel_graphic.label = _("0eV")


class MenuExampleExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.eels_analysis.menu_item_attach"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="~1.0")
        # be sure to keep a reference or it will be closed immediately.
        self.__measure_zlp_fit_spline_menu_item_ref = api.create_menu_item(MeasureZLPFitSplineMenuItemDelegate(api))
        self.__measure_zlp_count_pixels_menu_item_ref = api.create_menu_item(MeasureZLPCountPixelsMenuItemDelegate(api))
        self.__mark_0eV_menu_item_ref = api.create_menu_item(Mark0eVMenuItemDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__measure_zlp_fit_spline_menu_item_ref.close()
        self.__measure_zlp_fit_spline_menu_item_ref = None
        self.__measure_zlp_count_pixels_menu_item_ref.close()
        self.__measure_zlp_count_pixels_menu_item_ref = None
        self.__mark_0eV_menu_item_ref.close()
        self.__mark_0eV_menu_item_ref = None
