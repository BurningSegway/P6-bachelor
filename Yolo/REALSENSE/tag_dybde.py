import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
img = np.zeros((640, 480, 1), dtype=float)
# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

while True:
    Do_it = input("skal jeg tage et billede")


    if Do_it == "ja":
        # Create a context object. This object owns the handles to all connected realsense devices

        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        for x in range(640):
            for y in range(480):
                img[x,y,0]=depth.get_distance(x, y)

        #print(img)

        cv2.imshow('depth_colormap_rgb', img)
        cv2.waitKey(1)


        if not depth:
            print("ting")
            continue
        colorized_depth = np.asanyarray(depth.get_data())
        print(colorized_depth.shape)
        colorized_depth_rgb = cv2.cvtColor(colorized_depth, cv2.COLOR_BGR2RGB)
        cv2.imshow('depth_colormap_rgb', colorized_depth_rgb)
        cv2.waitKey(1)

        #cv2.imshow("depth", depth)
        #cv2.waitKey(1)








#except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
