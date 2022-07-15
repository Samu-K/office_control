"""
The main script that controls all functionality
Goal is that if this script is running, all functions work and are online
Depencies should be kept to a minimun
    from outside sources
    internal factoring should be done as much as possible
"""
# imports
import os
from datetime import datetime
    
# create function for easy log upkeep
def log_update(type: int, log_path="log.csv"):
    """
    Updates log and creates it if needed

    Args:
        type (int): 0 if startup, 1 if shutdown
        log_path (str, optional): File path to log. Defaults to "log.csv".
    """

    # fetch current time and date
    dt = datetime.now()
    date = dt.strftime("%d.%m.%Y")
    time = dt.strftime("%H:%M:%S")
        
    # check if log file exists
    if os.path.exists("log.csv"):
        # file exists
        with open("log.csv","a") as f:
            f.write("\n")
            f.write(f"{type}, {date}, {time} UTC+2")
            f.close()
    else:
        # file does not exist
        # create it and write data
        with open("log.csv","w") as f:
            f.write("type, date, time")
            f.write("\n")
            f.write(f"{type}, {date}, {time} UTC+2")
            f.close()

# update log on startup 
log_update(0)

# setup object detection
# setup cam

def presence_detection():
    # detects if someone is in room
    # returns state
    pass

def gesture_wake_up():
    # looks for gesture_detection wake up signal
    pass

def gesture_detection():
    # looks for certain gestures
    pass

