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
      e = Emitter(id=t.id+":count", name="Count Visible", targetId=rootRect.id, serviceId=client.clientId, schema={"type": "object", "properties": {"value": {"type": "number"}}})
      targetIds.append(t.id)
      t.emitterIds.append(e.id)
      client.saveTarget(t)
      client.setTargetStatus(t, instance, Status.Online)
      client.saveEmitter(e)

  rootRect.subTargets = targetIds
  client.saveTarget(rootRect)
  client.setTargetStatus(rootRect, instance, Status.Online)

def detect():
  videoPath = os.getenv("VIDEO_PATH")
  modelPath = os.getenv("MODEL_PATH")

  parentDir = Path(__file__).parent.parent.absolute()
  modelPath = parentDir.as_posix() + '/' + modelPath

  print("loading model")

  model = YOLO(modelPath)
  model.to('cuda')

  print("loading video")

  cap = cv2.VideoCapture(videoPath)

  if not cap.isOpened():
      print("Cannot open camera")
      exit()
  while True:
    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video
        continue

    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        results.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    counts = Counter(labels)

    rectId = 'root'
    for key, value in counts.items():
      targetId = rectId + ":" + key
      emitterId = targetId + ":count"
      client.pulseEmitter(emitterId=emitterId,data={"value": value})
    

    cv2.imshow("yolov8", annotated_image)

    if cv2.waitKey(1) == 27:
        break

def main():
  load_dotenv()

  print("starting detection")

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


if __name__ == '__main__':
  main()

