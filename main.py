import cv2
from djitellopy import tello
import cvzone

thres = 0.67
nmsThres = 0.5
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

pos_x = 0
pos_y = 20

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

me.takeoff()
me.move_up(pos_y)


center_x1 = 0
center_y1 = 0

while True:
    # success, img = cap.read()

    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2

            if center_x1 == 0:
                center_x1 = center_x
                center_y1 = center_y
            else:
                diff_x = center_x1 - center_x
                diff_y = center_y1 - center_y
                pos_y += diff_y
                print(diff_x,diff_y)
                # me.move_up(pos_y)
            if diff_x > 0:
                pass
                # me.rotate_clockwise(diff_x)
            else:
                pass
                # me.rotate_counter_clockwise(diff_x)

            print(center_x, center_y)
            # cv2.putText(img,"X",(center_x,center_y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 255, 0), 2)

            cvzone.cornerRect(img, box, rt = 3, )
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)

            cvzone.cornerRect(img, box)

    except:
        pass

    me.send_rc_control(0,0,0,0)
    cv2.imshow("image", img)
    cv2.waitKey(1)