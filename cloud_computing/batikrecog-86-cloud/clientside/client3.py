import requests
import json
import cv2
import base64
from imageio import imread
from requests.models import Response

addr = 'https://batikrecog-b21-cap0068.et.r.appspot.com/'
# addr = 'http://127.0.0.1:5000/'
test_url = addr + '/predict/pcpy'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

filename = "tesbatik1.jpg"
with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = base64.b64encode(data)

kirim = {'data'}

response = requests.post(test_url, data=b64_bytes, headers=headers)

print(response.text)

hasil = json.loads(response.text)
hasil = hasil["prediction"]

print("HASIL NYA SEBAGAI BERIKUT = ")
print(hasil)

