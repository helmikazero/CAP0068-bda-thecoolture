import requests
import json
import cv2
import base64
from imageio import imread
from requests.models import Response

# addr = 'https://batikrecog-bangkit-1.herokuapp.com/'
addr = 'http://127.0.0.1:8080/'
test_url = addr + '/predict/pcpy'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

filename = "batik_ceplok.jpg"
with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = base64.b64encode(data)

response = requests.post(test_url, data=b64_bytes, headers=headers)

hasil = json.loads(response.text)
hasil = hasil["prediction"]

print("HASIL NYA SEBAGAI BERIKUT = ")
print(hasil)

