import requests
import json
import cv2
import base64
from imageio import imread
from requests.models import Response
import io

addr = 'https://batikrecog-bangkit-1.herokuapp.com/'
# addr = 'http://127.0.0.1:5000/'
test_url = addr + '/predict/full'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

filename = "imstring_1.txt"
with open(filename, "rb") as fid:
    data = fid.read()

b64_bytes = data

img = imread(io.BytesIO(base64.b64decode(b64_bytes)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite('decode_result.jpg',img)

response = requests.post(test_url, data=b64_bytes, headers=headers)

hasil = json.loads(response.text)
hasil = hasil["prediction"]

print("HASIL NYA SEBAGAI BERIKUT = ")
print(hasil)

