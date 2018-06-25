import urllib3
import json
from bs4 import BeautifulSoup

img_base_location = 'data/images/'
data_sets = ['train', 'validation']

# Make sure you have the directories: data/images/test, data/images/train and data/images/validation
# for set in data_sets:
#     with open('data/{}.json'.format(set)) as json_data:
#         data = json.load(json_data)
#     images = data['images']
#     n_images = len(images)
#     print('{} images in the {} set'.format(n_images, set))
#     failures = 0
#     for image in images:
#         print('Downloading image {}/{}'.format(image['imageId'], n_images))
#         with urllib.request.urlopen(image['url']) as response, open(img_base_location + set + '/{}.jpg'.format(image['imageId']), 'wb') as file:
#             # print(response.getcode())
#             print(type(response))
#             print(response.getcode())
#             if response.getcode() == 200:
#                 file.write(response.read())
#             else:
#                 failures += 1

http = urllib3.PoolManager()
url = 'https://www.linkedin.com/in/brianwesterweel/'
response = http.request('GET', url)
print(response.data)

# print(profile)