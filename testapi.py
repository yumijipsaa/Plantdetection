import requests
img1=r'D:\itconv\Detection(Python)\Detection_models\test_images_paprika\20200428_111352.jpg'
img2=r'D:\itconv\Detection(Python)\Detection_models\test_images_paprika\20200428_111358.jpg'

files = {'paprika1': open(img1,'rb'),
         'paprika2': open(img2,'rb')}
data = {'model_type':'paprika'}

url = r'http://192.168.0.3:5000/plant-inference'

response = requests.post(url,data=data,files=files)
