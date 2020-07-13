import os
import sys
import urllib.request
from tqdm import tqdm
from itertools import chain

def download_nyctaxi_data(years, path):
  taxi_years = [
    "2014",
    "2015",
    "2016"
  ]

  if not set(years) <= set(taxi_years):
    print(years)
    print("years list not valid, please specify a sublist of")
    print(taxi_years)
    raise Exception("{years} list is not valid".format(years=years))

  data_dir = os.path.abspath(os.path.join(path, "nyctaxi"))
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  filenames = []
  local_paths = []
  for year in years:
    if year == "2016":
      start = 1
      end = 7
    else:
      start = 1
      end = 13
    if not os.path.exists(os.path.join(data_dir, year)):
      os.makedirs(os.path.join(data_dir, year))
    for i in range(start, end):
      filename = "yellow_tripdata_{year}-{month:02d}.csv".format(year=year, month=i)
      filenames.append(filename)
      local_path = os.path.join(data_dir, year, filename)
      local_paths.append(local_path)

  for year in years:
    for idx, filename in enumerate(filenames):
      filename_elements = [filename_element.split('-') for filename_element in filename.split('_')]
      filename_elements = list(chain.from_iterable(filename_elements))
      if year in filename_elements:
        url = "https://storage.googleapis.com/anaconda-public-data/nyc-taxi/csv/{year}/".format(year=year) + filename
        print("- Downloading " + url)
        if not os.path.exists(local_paths[idx]):
          with open(local_paths[idx], 'wb') as file:
            with urllib.request.urlopen(url) as resp:
              length = int(resp.getheader('content-length'))
              blocksize = max(4096, length // 100)
              with tqdm(total=length, file=sys.stdout) as pbar:
                while True:
                  buff = resp.read(blocksize)
                  if not buff:
                    break
                  file.write(buff)
                  pbar.update(len(buff))
        else:
          print("- File already exists locally")

  print("-------------------")
  print("-Download complete-")
  print("-------------------")
