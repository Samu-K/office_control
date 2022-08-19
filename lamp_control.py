import requests
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

class Lamp():
    """
    Used to control any Philips Hue lamp
    """
    def __init__(self, lamp_id:str, bridge_id:str, username:str):
        """
        Args:
            lamp_id (str): id of lamp to be controlled
            bridge_id (str): ip of the Hue Bridge that lamp is connected to
            username (str): username of developer to access lamps
        """
        # Setup the web address for lamp control
        self.address = f"https://{bridge_id}/clip/v2/resource/light/{lamp_id}"
        self.username = username
        
    def power(self,state: bool):
        """
        Turn lamp on or off
        Args:
            state (bool): State to set lamp to True = on and False = off
        """
        # Send a put request to lamp
        # Sets the lamps state to given state
        requests.put(
            url=self.address,
            headers={"hue-application-key": self.username},
            json={"on": {"on": state}},
            verify=False
        )
    
    def brightness(self, level: float):
        """
        Sets the lamp brightness to given level
        Args:
            level (float): level of brightness 0-100
        """
        if (level < 0) or (level > 100):
            raise ValueError("Input should be within 0-100")
        
        # Send put request to lamp
        # This changes to brightness to given level
        requests.put(
            url=self.address,
            headers={"hue-application-key": self.username},
            json={"dimming": {"brightness":level}},
            verify=False
        )