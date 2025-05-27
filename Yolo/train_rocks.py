####Old version of the code:


#from ultralytics import YOLO, checks, hub
#checks()

#hub.login('ultralytics_api_key_here')

#model = YOLO('ultralytics_project_link_here')
#results = model.train(device='mps') #i am uncertin weather the "device='msp'" is nececarry

####

from ultralytics import YOLO, checks, hub
checks()

hub.login('cec9a2ca8516f54e0441992802972efc4950a2b3bb') #use your own login

model = YOLO('https://hub.ultralytics.com/models/mTvD3VLZSTGEuweBOO3f') #use your own project link
results = model.train()
