
from openni import openni2
import numpy as np
import cv2
 
openni2.initialize(r"C:\OpenNI2\Bin")
 
dev = openni2.Device.open_any()
 
depth_stream = dev.create_depth_stream()
depth_stream.start()
 
print("Depth stream started... Press Q to quit")
 
cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
 
while True:
 
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
 
    depth_image = np.frombuffer(frame_data, dtype=np.uint16)
    depth_image = depth_image.reshape((480, 640))
 
    # ── FIX 1: Filter invalid zeros before normalizing ──
    valid_mask = depth_image > 0
 
    # ── FIX 2: Normalize only valid pixels ──
    depth_display = cv2.normalize(
        depth_image,
        None,
        0, 255,
        cv2.NORM_MINMAX
    )
    depth_display = np.uint8(depth_display)
 
    # ── FIX 3: Apply colormap for clear visualization ──
    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
 
    # Mask invalid pixels as black
    depth_colormap[~valid_mask] = 0
 
    # ── FIX 4: Show center distance ──
    cx, cy = 320, 240
    region = depth_image[cy-5:cy+5, cx-5:cx+5]
    valid_region = region[region > 0]
 
    if len(valid_region) > 0:
        center_dist = int(np.mean(valid_region))
        cv2.putText(
            depth_colormap,
            f"Center: {center_dist} mm ({center_dist/10:.1f} cm)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
 
    # Draw crosshair at center
    cv2.drawMarker(depth_colormap, (cx, cy), (255,255,255),
                   cv2.MARKER_CROSS, 20, 2)
 
    cv2.imshow("Depth", depth_colormap)
 
    if cv2.waitKey(1) == ord('q'):
        break
 
depth_stream.stop()
openni2.unload()
cv2.destroyAllWindows()