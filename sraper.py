import urllib.request



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
