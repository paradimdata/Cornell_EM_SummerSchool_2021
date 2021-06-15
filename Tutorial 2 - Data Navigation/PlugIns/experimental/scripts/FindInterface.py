import numpy
from nion.utils import Geometry
from nion.utils import Registry

def find_interface(interactive, api):
    # first grab the stem controller object by asking the Registry
    stem_controller = Registry.get_component("stem_controller")

    # get the scan controller and stop it from playing
    scan_controller = stem_controller.scan_controller
    scan_controller.stop_playing()
    stem_controller.probe_position = Geometry.FloatPoint(x=0.5, y=0.5)

    # establish the EELS camera object and stop it if it is playing
    eels_camera = stem_controller.eels_camera
    eels_camera.start_playing()

    # search will start at left and right edges of the data
    left = 0.0
    right = 1.0

    # loop until we are canceled or meet our criteria
    while not interactive.cancelled:
        print(f"Range left {left} right {right}")

        # this is the list of peak values in the data, representing thickness of the sample
        # larger value indicates less thickness (vacuum)
        peaks = list()

        # we can specify the number of steps, more steps will do a better job of finding vacuum
        steps = 4
        for i in range(steps):
            if interactive.cancelled:
                break
            # configure probe position (along center-y, going left-to-right)
            probe_position = 0.5, i / (steps - 1) * (right - left) + left
            print(f"settings probe position {probe_position}")
            # set the probe position
            stem_controller.probe_position = Geometry.FloatPoint.make(probe_position)
            # throw away an image; ensure exposure represents new scan position
            eels_camera.grab_next_to_start()
            # grab real data
            eels_data = eels_camera.grab_next_to_start()[0]
            # find the max value and put it into the peaks list
            peaks.append(numpy.amax(eels_data))
            # sleep here for better visibility of algorithm
            # time.sleep(1.0)

        if not interactive.cancelled:
            print(f"peaks {peaks}")

            # try to reduce our search space

            # first find the min and max values. max value will be vacuum.
            min_v = numpy.amin(peaks)
            max_v = numpy.amax(peaks)
            half_v = (min_v + max_v) * 0.5

            # find the index of the maximum (vacuum)
            mx = numpy.argmax(peaks)

            # now search to the left and right and find a value above the threshold
            # mark this as the maximum index
            mn = None
            for i in range(mx + 1, steps):
                if mn is None and peaks[i] < half_v:
                    mn = i
            for i in range(mx - 1, -1, -1):
                if mn is None and peaks[i] < half_v:
                    mn = i

            # now search to the left and right and find a value back below the threshold
            # mark this as the minimum index
            mx = None
            for i in range(mn + 1, steps):
                if mx is None and peaks[i] > half_v:
                    mx = i
            for i in range(mn - 1, -1, -1):
                if mx is None and peaks[i] > half_v:
                    mx = i

            # swap if necessary to make sure mn, mx are in order.
            if mn > mx:
                mn, mx = mx, mn

            print(f"mn {mn}  mx {mx}")

            # have we narrowed the search? if not break out. we're stuck.
            if mn == 0 and mx == steps - 1 or mn == mx:
                break

            # narrow the search
            left = mn / (steps - 1) * (right - left) + left
            right = mx / (steps - 1) * (right - left) + left

            # check criteria and break if done.
            if right - left < 0.001:
                break

    print("finished")


def script_main(api_broker):
    interactive = api_broker.get_interactive(version="1")
    interactive.print_debug = interactive.print
    api = api_broker.get_api(version="~1.0")
    find_interface(interactive, api)
