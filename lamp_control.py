import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

class Lamp():
    def __init__(self, lamp_id, bridge_id, username):
        self.address = f"https://{bridge_id}/clip/v2/resource/light/{lamp_id}"
        self.username = username
        
    def power(self,state):
        requests.put(
            url=self.address,
            headers={"hue-application-key": self.username},
            json={"on": {"on": state}},
            verify=False
        )
    
    def brightness(self, level):
        requests.put(
            url=self.address,
            headers={"hue-application-key": self.username},
            json={"dimming": {"brightness":level}},
            verify=False
        )