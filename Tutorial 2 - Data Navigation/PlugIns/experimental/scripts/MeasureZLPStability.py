from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd

from nion.eels_analysis import ZLP_Analysis

import functools
import numpy
import operator
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.stats


def sequence_estimate_zlp_pos(s):
    zs = numpy.empty((s.data_shape[0], ), numpy.double)
    for i in range(s.data_shape[0]):
        d = s[i].data
        a, zs[i], l, r = ZLP_Analysis.estimate_zlp_amplitude_position_width_com(d)
    # replace nan's with average
    mask = numpy.isnan(zs)
    zs[mask] = numpy.interp(numpy.flatnonzero(mask), numpy.flatnonzero(~mask), zs[~mask])
    return DataAndMetadata.new_data_and_metadata(zs, intensity_calibration=s.dimensional_calibrations[-1], dimensional_calibrations=[s.dimensional_calibrations[0]])


def convert_collection_to_sequence(c):
    c_dim_count = c.collection_dimension_count
    d_dim_count = c.datum_dimension_count
    assert not c.is_sequence
    assert 1 <= c_dim_count <= 3 and 1 <= d_dim_count <= 2
    c_shape = functools.reduce(operator.mul, [c.data_shape[ci] for ci in c.collection_dimension_indexes])
    data_descriptor = DataAndMetadata.DataDescriptor(True, 0, d_dim_count)
    ic = c.intensity_calibration
    dc = [c.collection_dimensional_calibrations[-1]] + c.datum_dimensional_calibrations
    return DataAndMetadata.new_data_and_metadata(numpy.reshape(c.data, (c_shape, ) + c.data_shape[-d_dim_count:]), intensity_calibration=ic, dimensional_calibrations=dc, data_descriptor=data_descriptor)


def measure_relative_zlp_pos(interactive, api):
    print("Starting measure relative ZLP position...")

    relative = True

    # first find the target SI from which to measure shifts
    src_data_item = api.application.document_windows[0].target_data_item
    src = src_data_item.xdata

    # measure the zlp positions, treating the collection as a sequence
    zs = sequence_estimate_zlp_pos(convert_collection_to_sequence(src))

    # subtract the mean
    if relative:
        zs2_data = zs.data - numpy.mean(zs.data)
    else:
        zs2_data = zs.data

    # grab the exposure time if possible.
    exposure = src_data_item.get_metadata_value("stem.camera.exposure_s") if src_data_item.has_metadata_value("stem.camera.exposure_s") else None
    if not exposure:
        exposure = interactive.get_integer("Exposure time (ms):") / 1000
    dimensional_calibrations = [Calibration.Calibration(scale=exposure, units="s")]
    if relative:
        intensity_calibration = Calibration.Calibration(scale=zs.intensity_calibration.scale, units=zs.intensity_calibration.units)
    else:
        intensity_calibration = zs.intensity_calibration

    # construct the data item
    zs2 = DataAndMetadata.new_data_and_metadata(zs2_data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
    data_item = api.library.create_data_item_from_data_and_metadata(zs2)

    # and show the data item
    api.application.document_windows[0].display_data_item(data_item)

    print("Finished measure shifts...")



def measure_shifts(interactive, api):
    print("Starting measure shifts...")

    # first find the target SI from which to measure shifts
    src_data_item = api.application.document_windows[0].target_data_item
    src = src_data_item.xdata

    # save the x calibration; and energy calibration
    xc = src.dimensional_calibrations[1]
    ec = src.dimensional_calibrations[2]

    # DISABLED: threshold the data (10% of max). This removes the tails of the ZLP and makes cross correlation inaccurate.
    src_data = src.data
    # src_data_max = numpy.amax(src_data)
    # src_data = scipy.stats.threshold(src_data, src_data_max * 0.1, src_data_max, 0)

    # apply a hann function to de-emphasize the edges
    src_data *= scipy.signal.hann(src_data.shape[-1])

    # convert the SI to a sequence ordered by x- and the y- direction
    data_descriptor = DataAndMetadata.DataDescriptor(True, 0, 1)
    src_seq = DataAndMetadata.new_data_and_metadata(numpy.reshape(src_data, (src.data_shape[0] * src.data_shape[1], src.data_shape[-1])), data_descriptor=data_descriptor)

    # from the sequence, measure the relative shift between each successive frame
    m = xd.sequence_register_translation(src_seq, 100)

    # measuring shift will produce a n x 1 array; squeeze the data to get rid of the unnecessary dimension
    md = numpy.squeeze(m.data)

    # filter out spikes
    md_median = scipy.signal.medfilt(md, kernel_size=5)
    md_min = numpy.amin(md_median)
    md_max = numpy.amax(md_median)
    md_range = md_max - md_min
    t_min = md_min - md_range * 0.1
    t_max = md_max + md_range * 0.1
    md = scipy.stats.threshold(md, t_min, t_max, 0)

    # since sequence_register measures the shift between each frame, integrate it to produce absolute shift
    # and also note that register_translation produces negative values for historical/compatibility reasons (scikit).
    md = numpy.cumsum(-md)

    # grab the exposure time if possible.
    exposure = src_data_item.get_metadata_value("stem.camera.exposure_s") if src_data_item.has_metadata_value("stem.camera.exposure_s") else None

    if not exposure:
        exposure = interactive.get_integer("Exposure time (ms):") / 1000

    # provide calibrations if possible
    dimensional_calibrations = [Calibration.Calibration(scale=exposure, units="s")]
    intensity_calibration = Calibration.Calibration(scale=ec.scale, units=ec.units)

    # construct the data item
    xmd = DataAndMetadata.new_data_and_metadata(md, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations)
    data_item = api.library.create_data_item_from_data_and_metadata(xmd)

    # and show the data item
    api.application.document_windows[0].display_data_item(data_item)

    print("Finished measure shifts...")


def script_main(api_broker):
    interactive = api_broker.get_interactive(version="1")
    api = api_broker.get_api(version="1")
    measure_relative_zlp_pos(interactive, api)
    # measure_shifts(interactive, api)
