from naoqi import ALProxy

# To get the constants relative to the video.
import vision_definitions
from PIL import Image

class Nao(object):
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        
        self._cameraID = 0
        
        self._registerImageClient()
    
    def disconnect(self):
        print 'disconnect'
        self._unregisterImageClient()
            
    def get_image(self):
        # Get a camera image.
        # image[6] contains the image data passed as an array of ASCII chars.
        naoImage = self._videoProxy.getImageRemote(self._imgClient)
        
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        
        # Create a PIL Image from our pixel array.
        im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
        
        return im
           
    def _registerImageClient(self):
        """
        Register our video module to the robot.
        """
        print 'connect video module to robot'
        self._videoProxy = ALProxy("ALVideoDevice", self._ip, self._port)
        
        resolution = vision_definitions.kQVGA  # 320 * 240
        colorSpace = vision_definitions.kRGBColorSpace
        
        self._imgClient = self._videoProxy.subscribe("_client", resolution, colorSpace, 5)

        # Select camera.
        self._videoProxy.setParam(vision_definitions.kCameraSelectID,
                                  self._cameraID)

    def _unregisterImageClient(self):
        """
        Unregister our naoqi video module.
        """
        if self._imgClient != "":
            self._videoProxy.unsubscribe(self._imgClient)
    