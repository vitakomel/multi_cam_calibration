
import cv2
import numpy as np
import depthai as dai
import contextlib
import blobconverter
from matplotlib import pyplot as plt
from matplotlib import cm


# Chessboard parameters
chessboard_width = 6
chessboard_height = 9
square_size = 25.2

label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# Colors for plot
viridis = cm.get_cmap('viridis', len(label_map))(range(0, len(label_map)))


def find_chessboard(frame, chessboard_params):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_found, corners = cv2.findChessboardCorners(gray_frame, chessboard_params)

    if chessboard_found:
        corners = cv2.cornerSubPix(gray_frame, corners, (11,11), (-1,-1), criteria)

    return chessboard_found, corners


# World coordinates for chessboard
def get_grid(height, width, square_size):
	obj_points = np.zeros((width * height, 3), np.float32)
	obj_points[:, :2] = np.mgrid[0:height, 0:width].T.reshape(-1,2)

	return square_size * obj_points


# Frames from all devices
def get_frames(output_queues):
    frames = {}

    for id, q in output_queues.items():
        frames[id] = {
            'rgb': q['rgb'].get().getCvFrame(),
            'nn': q['nn'].get().detections
        }
                 
    return frames


def convert_to_pixels(point, cal):
    x = point[0] * cal[0, 0] / point[2] + cal[0, 2]
    y = point[1] * cal[1, 1] / point[2] + cal[1, 2]

    return (int(x), int(y))


def transform(point, rot, t):
    transformed_point = rot @ point + t

    return transformed_point


def inv_transform(point, rot, t):
    transformed_point = transform(point, rot.T, -rot.T @ t)

    return transformed_point


# Visualize axes on frames
def visualize_axes(frame, cal, rot, t):
    base_point = transform(np.array([[0], [0], [0]]), rot, t)
    base_point_pixels = convert_to_pixels(base_point, cal)

    x_end = transform(np.array([[200], [0], [0]]), rot, t)
    x_end_pixels = convert_to_pixels(x_end, cal)

    y_end = transform(np.array([[0], [200], [0]]), rot, t)
    y_end_pixels = convert_to_pixels(y_end, cal)

    z_end = transform(np.array([[0], [0], [200]]), rot, t)
    z_end_pixels = convert_to_pixels(z_end, cal)

    cv2.line(frame, base_point_pixels, x_end_pixels, (0, 0, 255), 2)
    cv2.line(frame, base_point_pixels, y_end_pixels, (0, 0, 255), 2)
    cv2.line(frame, base_point_pixels, z_end_pixels, (0, 0, 255), 2)

    cv2.putText(frame, "x", x_end_pixels, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
    cv2.putText(frame, "y", y_end_pixels, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
    cv2.putText(frame, "z", z_end_pixels, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))


# Create pipeline
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth = pipeline.create(dai.node.StereoDepth)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.setDepthAlign(dai.CameraBoardSocket.RGB)
depth.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(False)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
mono_left.out.link(depth.left)
mono_right.out.link(depth.right)

detector = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
detector.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detector.setConfidenceThreshold(0.75)
cam_rgb.preview.link(detector.input)
depth.depth.link(detector.inputDepth)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName('rgb')
detector.passthrough.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName('nn')
detector.out.link(xout_nn.input)


with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()
    intrinsics = {}
    output_queues = {}

    for device_info in device_infos:

        device = stack.enter_context(dai.Device(pipeline))
        mxid = device.getMxId()

        calibration_handler = device.readCalibration()
        intrinsics_rgb = np.array(calibration_handler.getCameraIntrinsics(dai.CameraBoardSocket.RGB, resizeWidth=300, resizeHeight=300))
        intrinsics[mxid] = intrinsics_rgb

        output_queues[mxid] = {
            'rgb': device.getOutputQueue(name='rgb', maxSize=4, blocking=False),
            'nn': device.getOutputQueue(name='nn', maxSize=4, blocking=False)
        }


    obj_points = get_grid(chessboard_height, chessboard_width, square_size)
    num_calibrated_devices = 0
    
    # Calibrate devices
    while num_calibrated_devices < len(device_infos):

        frames = get_frames(output_queues)
        num_calibrated_devices = 0
        transformations = {}

        for id, frame in frames.items():
            chessboard_found, corners = find_chessboard(frame['rgb'], (chessboard_height, chessboard_width))

            if not chessboard_found:
                print('Move the chessboard.')
                break

            _, rot, t = cv2.solvePnP(obj_points, corners, intrinsics[id], np.zeros((4, 1)), flags=0)
            transformations[id] = {'rot': cv2.Rodrigues(rot)[0], 't': t}
            num_calibrated_devices += 1


    print('Calibration is completed.')

    fig = plt.figure()
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    old_points = []
    ann_list = []

    while True:

        frames = get_frames(output_queues)
        all_detections = []

        for id, frame in frames.items():
            
            height = frame['rgb'].shape[0]
            width  = frame['rgb'].shape[1]

            for detection in frame['nn']:

                x1 = int(detection.xmin * width)
                y1 = int(detection.ymin * height)
                x2 = int(detection.xmax * width)
                y2 = int(detection.ymax * height)
                cv2.rectangle(frame['rgb'], (x1, y1), (x2, y2), (255, 0, 0))

                x = detection.spatialCoordinates.x
                y = detection.spatialCoordinates.y
                z = detection.spatialCoordinates.z
                point = np.array([[x], [-y], [z]])
                
                # Transform point to the world coordinate system
                point = inv_transform(point, transformations[id]['rot'], transformations[id]['t'])
                all_detections.append({'id': id, 'label': detection.label, 'point': point})

                try:
                    label = label_map[detection.label]
                except:
                    label = detection.label
                
                cv2.putText(frame['rgb'], str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame['rgb'], "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame['rgb'], f'X: {int(point[0])} mm', (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame['rgb'], f'Y: {int(point[1])} mm', (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame['rgb'], f'Z: {int(point[2])} mm', (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            visualize_axes(frame['rgb'], intrinsics[id], transformations[id]['rot'], transformations[id]['t'])
            cv2.imshow(id, frame['rgb'])

        # Change colors
        if len(old_points) > 5:
            points = old_points.pop(0)
            for p in points:
                plt.scatter(p[0], p[1], color='silver', s=500)

        # Remove annotations
        for ann in ann_list:
            ann.remove()

        ann_list[:] = []
        new_old_points = []

        # Find corresponding detections
        for i in range(len(all_detections)):

            det_i = all_detections[i]
            matches = [det_i['point']]

            for j in range(i, len(all_detections)):
                det_j = all_detections[j]

                # If two detections with the same label are less than 20 cm apart, consider them as a match.
                if det_i['id'] != det_j['id'] and det_i['label'] == det_j['label'] and np.linalg.norm(det_i['point'] - det_j['point']) < 200:
                    matches.append(det_j['point'])

            # If a detection does not have a corresponding detection, do not plot it.
            if len(matches) > 1:
                # Average of all corresponding detections          
                avg = np.mean(np.array(matches), axis=0)
                plt.scatter(avg[0], avg[1], color=viridis[det_i['label'] % 30], s=500)
                new_old_points.append(avg[:2])

                try:
                    label = label_map[detection.label]
                except:
                    label = detection.label
                
                ann = plt.annotate(str(label), (avg[0], avg[1]))
                ann_list.append(ann)


        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (300, 300))
        cv2.imshow("plot", img)

        old_points.append(new_old_points)
            
        if cv2.waitKey(1) == ord('q'):
            break
