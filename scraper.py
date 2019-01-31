import urllib.request

from google_images_download import google_images_download


for i in range(1,9):
    url = "https://raw.githubusercontent.com/aaronice/tensorflow-animals/master/tf_files/animals/squirrel/pic_00"+ str(i) + ".jpg"
    filename = 'pic_00' + str(i) + '.jpg'
    response = urllib.request.urlretrieve(url, filename)

for i in range (10,99):
    url = "https://raw.githubusercontent.com/aaronice/tensorflow-animals/master/tf_files/animals/squirrel/pic_0"+ str(i) + ".jpg"
    filename = 'pic_0' + str(i) + '.jpg'
    response = urllib.request.urlretrieve(url, filename)

for i in range(100,516):
    try:
        url = "https://raw.githubusercontent.com/aaronice/tensorflow-animals/master/tf_files/animals/squirrel/pic_"+ str(i) + ".jpg"
        filename = 'pic_' + str(i) + '.jpg'
        urllib.request.urlretrieve(url, filename)
    except:
        urllib.error.HTTPError

response_1 = google_images_download.googleimagesdownload()

arguments = {"keywords": "dog","limit":500,"print_urls":True, "chromedriver" : "C:\\Users\\jguzm\\chromedriver\\chromedriver.exe"}

paths = response_1.download(arguments)

print(paths)
