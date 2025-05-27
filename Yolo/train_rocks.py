#from ultralytics import YOLO, checks, hub
#checks()

#hub.login('ultralytics_api_key_here')

#model = YOLO('https://hub.ultralytics.com/models/drsVaD3IRc5Cmx2HQQeW')
#results = model.train(device='mps')

from ultralytics import YOLO, checks, hub
checks()

hub.login('cec9a2ca8516f54e0441992802972efc4950a2b3bb')

model = YOLO('https://hub.ultralytics.com/models/mTvD3VLZSTGEuweBOO3f')
results = model.train()


