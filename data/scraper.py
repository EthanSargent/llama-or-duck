import urllib.request
import shutil
import requests

# large troves of llama and duck images 
DUCKS_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01846331"
LLAMAS_URL  = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02437616"

def crawl(url, folder_name):
	""" Gathers the list of .jpg urls at url, saves them in folder_name, and
		maintains an index file with the list of urls from which these images
		were downloaded.
	"""
	path_to_index = folder_name + "/" + folder_name + "-index"

	with urllib.request.urlopen(url) as response, open(path_to_index, 'wb') as out_file:
		shutil.copyfileobj(response, out_file)
		f = open(path_to_index, 'r+')
		image_number = 0
		dead_links = 0

		for line in f:
			try:
				# chop off the newline character
				image_url = line[0:-1]
				# save images of the form "llamas/llamas-55.jpg"
				path_to_image = folder_name + "/" + folder_name + "-" + str(image_number) + ".jpg"
				response = requests.get(image_url, stream=True)
				with open(path_to_image, 'wb') as out_file:
					shutil.copyfileobj(response.raw, out_file)
				del response
				image_number += 1
				print("\r" + image_number + " images downloaded. " + dead_links + " dead links.")

			except Exception:
				dead_links = dead_links + 1
				continue

# crawl(LLAMAS_URL, "llamas")
crawl(DUCKS_URL, "ducks")

response = requests.get("http://www.defenders.org/sites/default/files/styles/large/public/grizzly-bear-harry-bosen-dpc.jpg", stream=True)