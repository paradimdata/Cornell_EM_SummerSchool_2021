import numpy
import uuid
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.utils import Registry

def acquire_multi_eels(interactive, api):
    # first grab the stem controller object by asking the Registry
    stem_controller = Registry.get_component("stem_controller")

    # establish the EELS camera object and stop it if it is playing
    eels_camera = stem_controller.eels_camera
    eels_camera.stop_playing()

    print(eels_camera.hardware_source_id)

    # this table represents the acquisitions to be performed
    # each entry is energy offset, exposure (milliseconds), and the number of frames to integrate
    table = [
    # energy offset, exposure(ms), N frames
      (0, 100, 2),
      (10, 100, 2),
      #(250, 1000, 10),
      #(0, 100, 5),
    ]

    # this is the list of integrated spectra that will be the result of this script
    spectra = list()

    # this algorithm handles dark subtraction specially - so dark subtraction and gain normalization should
    # be disabled in the camera settings; this algorithm will handle dark subtraction itself.
    do_dark = True
    do_gain = False

    print("start taking data")

    energy_offset_control = "EELS_MagneticShift_Offset"  # for hardware EELS
    # energy_offset_control = "EELS_MagneticShift_Offset"  # for simulator

    tolerance_factor_from_nominal = 1.0
    timeout_for_confirmation_ms = 3000

    for energy_offset_ev, exposure_ms, frame_count in table:
        # for each table entry, set the drift tube loss to the energy offset
        stem_controller.SetValAndConfirm(energy_offset_control, energy_offset_ev, tolerance_factor_from_nominal, timeout_for_confirmation_ms)

        # configure the camera to have the desired exposure
        frame_parameters = eels_camera.get_current_frame_parameters()
        frame_parameters["exposure_ms"] = exposure_ms
        eels_camera.set_current_frame_parameters(frame_parameters)

        # disable blanker
        stem_controller.SetValAndConfirm("C_Blank", 0, tolerance_factor_from_nominal, timeout_for_confirmation_ms)

        # acquire a sequence of images and discard it; this ensures a steady state
        eels_camera.grab_sequence_prepare(frame_count)
        eels_camera.grab_sequence(frame_count)

        # acquire a sequence of images again, but now integrate the acquired images into a single image
        eels_camera.grab_sequence_prepare(frame_count)
        xdata = eels_camera.grab_sequence(frame_count)[0]

        print(f"grabbed data of shape {xdata.data_shape}")

        # extract the calibration info
        counts_per_electron = xdata.metadata.get("hardware_source", dict()).get("counts_per_electron", 1)
        exposure_ms = xdata.metadata.get("hardware_source", dict()).get("exposure", 1)
        intensity_scale = xdata.intensity_calibration.scale / counts_per_electron / xdata.dimensional_calibrations[-1].scale / exposure_ms / frame_count

        # now sum the data in the sequence/time dimension. use xd.sum to automatically handle metadata such as calibration.
        xdata = xd.sum(xdata, 0)

        # if dark subtraction is enabled, perform another similar acquisition with blanker enabled and subtract it
        if do_dark:
            # enable blanker
            stem_controller.SetValAndConfirm("C_Blank", 1, tolerance_factor_from_nominal, timeout_for_confirmation_ms)

            # acquire a sequence of images and discard it; this ensures a steady state
            eels_camera.grab_sequence_prepare(frame_count)
            eels_camera.grab_sequence(frame_count)

            # acquire a sequence of images again, but now integrate the acquired images into a single image
            eels_camera.grab_sequence_prepare(frame_count)
            dark_xdata = eels_camera.grab_sequence(frame_count)[0]

            # sum it and subtract it from xdata
            dark_xdata = xd.sum(dark_xdata, 0)
            xdata = xdata - dark_xdata

            print(f"subtracted dark data of shape {dark_xdata.data_shape}")
        if do_gain:
            # divide out the gain
            gain_uuid = uuid.uuid4()  # fill this in with the actual gain image uuid
            gain = interactive.document_controller.document_model.get_data_item_by_uuid(gain_uuid)
            if gain is not None:
                xdata = xdata / gain.xdata

        # next sum the 2d data into a 1d spectrum by collapsing the y-axis (0th dimension)
        # also configure the intensity calibration and title.
        spectrum = xd.sum(xdata, 0)
        spectrum.data_metadata._set_intensity_calibration(Calibration.Calibration(scale=intensity_scale, units="e/eV/s"))
        spectrum.data_metadata._set_metadata({"title": f"{energy_offset_ev}eV {int(exposure_ms*1000)}ms [x{frame_count}]"})

        # add it to the list of spectra
        spectra.append(spectrum)

    # disable blanking and return drift tube loss to 0.0eV
    stem_controller.SetValAndConfirm("C_Blank", 0, tolerance_factor_from_nominal, timeout_for_confirmation_ms)
    stem_controller.SetValAndConfirm(energy_offset_control, 0, tolerance_factor_from_nominal, timeout_for_confirmation_ms)

    print("finished taking data")

    # when multi display is available, we can combine the spectra into a single line plot display without
    # padding the data; but for now, we need to use a single master data item where each row is the same length.

    if len(spectra) > 0:
        # define the padded spectra list
        padded_spectra = list()

        # extract calibration info
        ev_per_channel = spectra[0].dimensional_calibrations[-1].scale
        units = spectra[0].dimensional_calibrations[-1].units
        min_ev = min([spectrum.dimensional_calibrations[-1].convert_to_calibrated_value(0) for spectrum in spectra])
        max_ev = max([spectrum.dimensional_calibrations[-1].convert_to_calibrated_value(spectrum.data_shape[-1]) for spectrum in spectra])

        # calculate what the length of the padded data will be
        data_length = int((max_ev - min_ev) / ev_per_channel)

        # for each spectra, pad it out to the appropriate length, putting the actual data in the proper range
        for spectrum in spectra:
            energy_offset_ev = int((spectrum.dimensional_calibrations[-1].convert_to_calibrated_value(0) - min_ev) / ev_per_channel)
            calibration_factor = spectrum.intensity_calibration.scale / spectra[0].intensity_calibration.scale
            data = numpy.zeros((data_length, ))
            data[energy_offset_ev:energy_offset_ev + spectrum.data_shape[-1]] = spectrum.data * calibration_factor
            padded_spectrum = DataAndMetadata.new_data_and_metadata(data, spectrum.intensity_calibration, [Calibration.Calibration(min_ev, ev_per_channel, units)])
            padded_spectra.append(padded_spectrum)

        # stack all of the padded data together for display
        master_xdata = xd.vstack(padded_spectra)

        # show the data
        window = api.application.document_windows[0]
        data_item = api.library.create_data_item_from_data_and_metadata(master_xdata)
        legends = [s.metadata["title"] for s in spectra]
        data_item.title = f"MultiEELS ({', '.join(legends)})"
        window.display_data_item(data_item)

    print("finished")


def script_main(api_broker):
    interactive = api_broker.get_interactive(version="1")
    interactive.print_debug = interactive.print
    api = api_broker.get_api(version="~1.0")
    acquire_multi_eels(interactive, api)
