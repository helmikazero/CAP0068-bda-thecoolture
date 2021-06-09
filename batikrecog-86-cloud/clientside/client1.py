import requests
import json
import cv2

addr = 'https://batikrecog-bangkit-1.herokuapp.com/'
#addr = 'http://127.0.0.1:5000/'
test_url = addr + '/predict'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('batik_ceplok.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
# print (jsonlib.loads(response.text))

#print(response)

hasil = json.loads(response.text)



print("HASIL NYA SEBAGAI BERIKUT = ")
print(hasil["label"])
# print(response.text)

# expected output: {u'message': u'image received. size=124x124'}