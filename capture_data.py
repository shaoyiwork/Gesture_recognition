import time
import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (200, 200)
    camera.start_preview()
    #warm up
    time.sleep(5)
    for i in range(200):
        camera.capture("hand_data/train/four/"+'four_'+str(i)+'.png')

