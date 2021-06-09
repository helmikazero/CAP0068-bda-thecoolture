import base64
from inspect import cleandoc
import io
import cv2
from imageio import imread
import os
import urllib.parse

filename = "tesbatik1.jpg"
with open(filename, "rb") as fid:
    data = fid.read()


def save_text(destring,txtname):
    if os.path.exists(txtname):
        os.remove(txtname)
    f = open(txtname, "x")
    f.write(destring)
    f.close

def open_text(txtname):
    f = open(txtname, "r")
    result = f.read()
    return result

b64_bytes = base64.b64encode(data)

save_text(str(b64_bytes),"b64_bytes.txt")


pdata0 = base64.b64decode(b64_bytes)
pdata1 = io.BytesIO(pdata0)
img = imread(pdata1)

cv2.imwrite("theimage.jpg",img)


# BEDA DUNIA

rdata = open_text("rdata.txt")

b64_bz = base64(rdata)

clean_data = urllib.parse.unquote(rdata)

save_text(clean_data,"clean_data.txt")