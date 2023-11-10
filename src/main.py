import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from supervision import ColorLookup
import os
from dotenv import load_dotenv
from pathlib import Path
import websocket
import rel
import socket
from datetime import datetime, timezone
from threading import Thread
from collections import Counter

import super_gradients

from coco import labels

from lib.registry import client
from lib.exec import Instance, InstanceStatus, Machine, Target, Emitter, Status
from lib.registry import client


def on_message(ws, message):
    client.parseMessage(message)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
  client.setSend(ws.send)
  print("Opened connection")

def make_machine() -> Machine: 
	machine = Machine(socket.gethostname())
	return machine

def on_exec_connected():
  print("Exec connected")

  serviceId="coco"
  clientId = client.clientId

  machine = make_machine()
  
  instance = Instance(
		id=machine.id+":"+serviceId, 
		name=serviceId, 
		serviceId=serviceId, 
		clientId=clientId, 
		serviceTypeCode="coco", 
		status=InstanceStatus.Available, 
		machineId=machine.id, 
    color="#61c0ff"
	)

  client.set(machine)
  client.set(instance)

  rootRect = Target(
    id="root", 
    name="root", 
    fgColor="#61c0ff", 
    serviceId=instance.serviceId, 
    category="Rectangle", 
    rootLevel=True, 
    actionIds=[], 
    emitterIds=[], 
    subTargets=[], 
    bgColor="#61c0ff", 
    lastUpdated=datetime.now(timezone.utc).isoformat()
  )
  targetIds = []
  
  for key, value in labels.items(): 
    cat = key
    for label in value: 
      t = Target(
        id=rootRect.id + ":" + label,
        name=label,
        fgColor="#61c0ff",
        bgColor="#61c0ff",
        emitterIds=[],
        serviceId=instance.serviceId,
        category=cat,
        rootLevel=False,
        actionIds=[],
        subTargets=[],
        lastUpdated=datetime.now(timezone.utc).isoformat(),
      )
      countVisible = Emitter(id=t.id+":count", name="Count Visible", targetId=rootRect.id, serviceId=instance.serviceId, schema={"type": "object", "properties": {"value": {"type": "number"}}})
      targetIds.append(t.id)
      t.emitterIds.append(countVisible.id)
      client.saveTarget(t)
      client.setTargetStatus(t, instance, Status.Online)
      client.saveEmitter(countVisible)

  poses = Target(
    id=rootRect.id + ':poses',
    name="Poses",
    fgColor="#61c0ff",
    bgColor="#61c0ff",
    emitterIds=[],
    serviceId=instance.serviceId,
    category="Pose",
    rootLevel=False,
    actionIds=[],
    subTargets=[],
    lastUpdated=datetime.now(timezone.utc).isoformat(),
  )

  posesEmitter = Emitter(
     id=poses.id+":poses", 
     name="Count Visible", 
     targetId=poses.id, 
     serviceId=instance.serviceId, 
     schema={"type": "object", "properties": {"value": {"type": "number"}}}
  )

  poses.emitterIds.append(posesEmitter.id)
  targetIds.append(poses.id)
  client.saveTarget(poses)
  client.saveEmitter(posesEmitter)
  client.setTargetStatus(poses, instance, Status.Online)

  rootRect.subTargets = targetIds
  client.saveTarget(rootRect)
  client.setTargetStatus(rootRect, instance, Status.Online)

def detect():
  videoPath = os.getenv("VIDEO_PATH")
  modelPath = os.getenv("MODEL_PATH")

  parentDir = Path(__file__).parent.parent.absolute()
  # modelPath = parentDir.as_posix() + modelPath

  print("loading model")
  print(modelPath)

  yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
  model = YOLO(modelPath)
  model.to('cuda')

  print("loading video")
  print(videoPath)

  cap = cv2.VideoCapture(videoPath)

  if not cap.isOpened():
      print("Cannot open camera")
      exit()

  showImage = os.getenv("SHOW_IMAGE")


  while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
        continue
    
    results = model(frame)[0]

    pose_predictions  = yolo_nas.predict(frame, conf=0.5)

    pose_prediction = pose_predictions[0].prediction # One prediction per image - Here we work with 1 image, so we get the first.

    if pose_prediction is not None:
      pose_prediction  = pose_prediction.poses    
       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
      client.pulseEmitter(emitterId="root:poses:poses", data={"value": pose_prediction.tolist()})

    detections = sv.Detections.from_ultralytics(results)

    labels = [
        results.names[class_id]
        for class_id
        in detections.class_id
    ]

    rectId = 'root'
    counts = Counter(labels)
    for key, value in counts.items():
      targetId = rectId + ":" + key
      emitterId = targetId + ":count"
      client.pulseEmitter(emitterId=emitterId,data={"value": value})

    if not showImage.lower() == "true":
      continue

      
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    cv2.imshow("yolov8", annotated_image)

    if cv2.waitKey(1) == 27:
        break

def main():
  load_dotenv()

  print("starting detection")

  try: 
    detect_thread = Thread(target=detect)
    detect_thread.start()

    # Rship Connection
    rshipAddress = os.getenv("RSHIP_ADDRESS")
    rshipPort = os.getenv("RSHIP_PORT")
    uri = "ws://" + rshipAddress + ":" + rshipPort + '/myko'
    ws = websocket.WebSocketApp(uri,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    client.onExecConnected(on_exec_connected)
    ws.run_forever(dispatcher=rel, reconnect=5)  # Set dispatcher to automatic reconnection, 5 second reconnect delay if connection closed unexpectedly
    rel.signal(2, rel.abort)  # Keyboard Interrupt
    rel.dispatch()

  except KeyboardInterrupt:
    print("Keyboard Interrupt")
    exit()



if __name__ == '__main__':
  main()

