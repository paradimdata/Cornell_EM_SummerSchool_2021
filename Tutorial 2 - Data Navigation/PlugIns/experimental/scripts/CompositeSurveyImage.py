"""
Example script to do acquire a composite survey image using stage shift.

The script uses the center 50% of the image, shifts the stage by the appropriate amount
in x, y directions, and stitches the resulting images together into a larger super image.

To use:
    Run Nion Swift and get a good image
    Set the defocus value to a large number (positive or negative) such as 500000nm.
    Ensure that the aperture is circular and centered.
    Ensure that the aperture is large enough so that the center 50% of the image is exposed through aperture.
    Decide how many images to acquire by setting the 'size' variable.
    Decide how to reduce the acquired data by setting  the 'reduce' variable.
    Run the script from command line or PyCharm or another suitable Python interpreter.

TODO: SShft.x, SShft.y add rotation; need to fix this in AS2.
TODO: The size of the camera output is hardcoded to 1024, 1024. It should read from the camera object.
TODO: Update composite image live during acquisition. Requires improvements to nionlib.
"""

import math
import numpy
import time

import nionlib

def acquire_composite_survey_image(size, rotation=0, scale=1, reduce=4, print_fn=None):
    print_fn = print_fn if print_fn is not None else lambda *args: None

    document_controller = nionlib.api.application.document_controllers[0]
    library = nionlib.api.library
    camera = nionlib.api.get_hardware_source_by_id("nionccd1010", "1")
    autostem = nionlib.api.get_instrument_by_id("autostem_controller", "1")

    shift_x_control_name = "SShft.x"
    shift_y_control_name = "SShft.y"

    # grab stage original location
    sx_m = autostem.get_control_output(shift_x_control_name)
    sy_m = autostem.get_control_output(shift_y_control_name)

    tv_pixel_angle_rad = autostem.get_control_output("TVPixelAngle")
    defocus = autostem.get_control_output("C10")

    print_fn("Acquiring composite survey image...")
    print_fn("stage starting position (um) ", sx_m * 1e6, sy_m * 1e6)
    print_fn("pixel angle (rad) ", tv_pixel_angle_rad)
    print_fn("defocus (nm) ", defocus * 1e9)

    image_size = 1024, 1024  # TODO: grab this from camera
    image_dtype = numpy.float32

    image_width_m = abs(defocus) * math.sin(tv_pixel_angle_rad * image_size[0])

    master_sub_area_size = 512, 512
    master_sub_area = (image_size[0]//2 - master_sub_area_size[0]//2, image_size[1]//2 - master_sub_area_size[1]//2), master_sub_area_size

    reduce = max(1, reduce // (512 / master_sub_area_size[0]))

    sub_area_shift_m = image_width_m * (master_sub_area[1][0] / image_size[0])

    sub_area = (master_sub_area[0][0]//reduce,master_sub_area[0][1]//reduce), (master_sub_area[1][0]//reduce,master_sub_area[1][1]//reduce)

    print_fn("image width (um) ", image_width_m * 1e6)
    master_data = numpy.empty((sub_area[1][0] * size[0], sub_area[1][1] * size[1]), image_dtype)
    print_fn("master size ", master_data.shape)

    try:
        for row in range(size[0]):
            for column in range(size[1]):
                delta_x_m, delta_y_m = sub_area_shift_m * (column - size[1]//2), sub_area_shift_m * (row - size[0]//2)
                print_fn("offset (um) ", delta_x_m * 1e6, delta_y_m * 1e6)
                start = time.time()
                # when used below, we use the rotation rotated by 180 degrees since we are moving the stage, not the
                # view. i.e. use -angle and subtract the delta's.
                rotated_delta_x_m = (math.cos(rotation) * delta_x_m - math.sin(rotation) * delta_y_m) / scale
                rotated_delta_y_m = (math.sin(rotation) * delta_x_m + math.cos(rotation) * delta_y_m) / scale
                print_fn("rotated offset (um) ", rotated_delta_x_m * 1e6, rotated_delta_y_m * 1e6)
                # set both values. be robust, retrying if set_control_output fails.
                attempts = 0
                while attempts < 4:
                    attempts += 1
                    try:
                        tolerance_factor = 0.02
                        autostem.set_control_output(shift_x_control_name, sx_m - rotated_delta_x_m, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                        autostem.set_control_output(shift_y_control_name, sy_m - rotated_delta_y_m, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                    except TimeoutError as e:
                        print("Timeout row=", row, " column=", column)
                        continue
                    break
                print_fn("Time", time.time() - start, " row=", row, " column=", column)
                supradata = camera.grab_next_to_start()[0]
                data = supradata.data[master_sub_area[0][0]:master_sub_area[0][0] + master_sub_area[1][0]:reduce, master_sub_area[0][1]:master_sub_area[0][1] + master_sub_area[1][1]:reduce]
                slice_row = row
                slice_column = column
                slice0 = slice(slice_row * sub_area[1][0], (slice_row + 1) * sub_area[1][0])
                slice1 = slice(slice_column * sub_area[1][1], (slice_column + 1) * sub_area[1][1])
                master_data[slice0, slice1] = data

        data_item = library.create_data_item_from_data(master_data, "Composite Survey")

        document_controller.display_data_item(data_item)
    finally:
        # restore stage to original location
        autostem.set_control_output(shift_x_control_name, sx_m)
        autostem.set_control_output(shift_y_control_name, sy_m)

# these measurements are determined by a line made from a feature before a shift to a feature after a
# shift. for instance, make a line starting on a feature. then add 100um to SShft.x and measure the
# length of the line and the angle. plug those in here.
rotation = math.radians(-23)
scale = 1.2

acquire_composite_survey_image(size=(5, 5), rotation=rotation, scale=scale, print_fn=print)
