"""
::

  Face averager

  Usage:
    averager.py [--rootpath=<folder>] [--blur] [--alpha] [--plot]
              [--width=<width>] [--height=<height>] [--debug] [--timestamp=<timestamp>]

  Options:
    -h, --help                 Show this screen.
    --rootpath=<folder>      Root folder of the project
    --blur                     Flag to blur edges of image [default: False]
    --alpha                    Flag to save with transparent background [default: False]
    --width=<width>            Custom width of the images/video [default: 500]
    --height=<height>          Custom height of the images/video [default: 600]
    --plot                     Flag to display the average face [default: False]
    --debug                    If true, use username javl
    --timestamp=<timestamp>    Timestamp
    --version                  Show version.
"""

from docopt import docopt
import os
import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import locator
import aligner
import warper
import blender
import os
import sqlite3
import io
import time

def list_imgpaths(imgfolder):
  for fname in sorted(os.listdir(imgfolder), reverse=True):
    if (fname.lower().endswith('.jpg') or
       fname.lower().endswith('.png') or
       fname.lower().endswith('.jpeg')):
      yield os.path.join(imgfolder, fname)

def sharpen(img):
  blured = cv2.GaussianBlur(img, (0, 0), 2.5)
  return cv2.addWeighted(img, 1.4, blured, -0.4, 0)

def load_image_points(path, size, fromDb=False):
  img = scipy.ndimage.imread(path)[..., :3]
  if fromDb:
    with con:
      cur = con.cursor()
      data = cur.execute("SELECT points FROM entries WHERE name=?", ([os.path.basename(path)]))
      points = convert_array(cur.fetchone()[0])
  else:
    points = locator.face_points(path)

  if len(points) == 0:
    return None, None
  else:
    return aligner.resize_align(img, points, size)


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sqlite3.register_converter("points", convert_array)

def averager(imgpaths, resultFolder, width=500, height=600, alpha=False, blur_edges=False, useDb=False):
  #startTime = time.time()
  size = (height, width)

  images = []
  point_set = []

  if useDb:
    #con = sqlite3.connect(os.path.join(alignedFolder, 'points.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES)

    with con:
      cur = con.cursor()
      for path in imgpaths:
        img = scipy.ndimage.imread(path)[..., :3]
        if img is not None:
          filename = os.path.basename(path)
          data = cur.execute('SELECT points FROM entries WHERE name=?', ([filename])).fetchone()
          if data:
            images.append(img)
            point_set.append(convert_array(data[0]))
  else:
    for path in imgpaths:
      img, points = load_image_points(path, size)
      if img is not None:
        images.append(img)
        point_set.append(points)

  ave_points = locator.average_points(point_set)
  num_images = len(images)
  result_images = np.zeros(images[0].shape, np.float32)
  for i in xrange(num_images):
    #print '{0} of {1}'.format(i+1, num_images)

    result_images += warper.warp_image(images[i], point_set[i],
                                       ave_points, size, np.float32)

    # Store all images
    result_image = np.uint8(result_images / (i+1))
    # mpimg.imsave(os.path.join(resultFolder, filenames[i]), result_image)
    mpimg.imsave(os.path.join(resultFolder, "{:05d}.jpg".format(i)), result_image)

  # result_image = np.uint8(result_images / num_images)
  #result_image = np.uint8(result_images)
  #mpimg.imsave(os.path.join(resultFolder, "result_raw.jpg"), result_image)
  result_image = np.uint8(result_images / num_images)
  mpimg.imsave(os.path.join(resultFolder, "result.jpg"), result_image)
  #print "Took: {}".format(time.time()-startTime)


def extract_face(imgPath, savePath, width=500, height=600, useDb=False):
  size = (height, width)
  filename = os.path.basename(imgPath)
  #print filename
  # path = "{}/captures/capture_{}.jpg".format(rootpath, timestamp)
  img, points = load_image_points(imgPath, size, fromDb=False)
  # Save points to database

  # Save image
  result_image = np.zeros(img.shape, np.float32)
  result_image += warper.warp_image(img, points, points, size, np.float32)
  result_image = np.uint8(result_image)
  mpimg.imsave(os.path.join(savePath, filename), result_image)

  if useDb:
    with con:
      cur = con.cursor()
      #print "insert using filename: ", filename
      cur.execute("INSERT INTO entries ('name', 'points') VALUES(?, ?)", (filename, points))

if __name__ == "__main__":
  # Get arguments from doc
  args = docopt(__doc__, version='Face Averager 1.0')

  userName = "lifefair"
  if args['--debug']:
    userName = "javl"

  dataFolder = "/Users/{}/Documents/of_v0.9.3_osx_release/apps/myApps/average-face/bin/data".format(userName)
  captureFolder = "{}/{}".format(dataFolder, "captures")
  alignedFolder = "{}/{}".format(dataFolder, "aligned")
  resultFolder = "{}/{}".format(dataFolder, "result")
  archiveFolder = "{}/{}".format(dataFolder, "archive")

  # First, remove old result files
  os.system("rm -f {}/*.jpg".format(resultFolder))
  # Are there more than 100 aligned faces? move the extras
  # The aligned images appear after processing, so remove one less than captures
  # to keep the amount in sync
  os.system("ls -r {} | tail -n +102 | xargs -I {{}} mv {}/{{}} {}/captures".format(captureFolder, captureFolder, archiveFolder))
  os.system("ls -r {} | tail -n +101 | xargs -I {{}} mv {}/{{}} {}/aligned".format(alignedFolder, alignedFolder, archiveFolder))

  useDb = True

  # Connect to database
  con = None
  if useDb:
    #con = sqlite3.connect("{}/points.sqlite".format(dataFolder), detect_types=sqlite3.PARSE_DECLTYPES)
    con = sqlite3.connect(os.path.join(dataFolder, 'points.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES)
    with con:
      cur = con.cursor()
      cur.execute('CREATE TABLE IF NOT EXISTS "main"."entries" \
        ("id" INTEGER PRIMARY KEY  AUTOINCREMENT  NOT NULL , \
        "name" TEXT, \
        "points" TEXT)')

  # Extract the new face and save it to the database
  extract_face(os.path.join(captureFolder, "{}.jpg".format(args['--timestamp'])), alignedFolder, int(args['--width']), int(args['--height']), useDb=True)

  ##if useDb:
  ##  with con:
  ##    cur = con.cursor()
  ##    data = cur.execute("SELECT * FROM ENTRIES ORDER BY id ASC")
  ##    points = data.fetchone()[2]
      ##print type(points)
      ##print type(convert_array(points))
    # el =  np.load(data.fetchone()[2])


  # Last, create a new average
  #averager(args['--rootpath'], list_imgpaths("{}/aligned".format(args['--rootpath'])), int(args['--width']),
  #        int(args['--height']), args['--alpha'], args['--blur'])
  averager(list_imgpaths(alignedFolder), resultFolder, int(args['--width']), int(args['--height']), args['--alpha'], args['--blur'], useDb=useDb)
