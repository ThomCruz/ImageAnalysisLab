#Original Author: Vinayak(2015ucp1057)
import cv2
import os
from os.path import isfile, join
import shutil

video_path = input('Enter Video File Path [Video.avi] ')
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 1
success = True

# create temporary directory to store frames
temp_directory = 'temp'
if not os.path.exists(temp_directory):
  os.makedirs(temp_directory)

# iterate over all frames
while success:
  cv2.imwrite("%s\\%.4d.jpg" % (temp_directory, count), image)
  success, image = vidcap.read()
  count += 1

fps_multiplier = float(input('Enter Frame Rate Multiplier [2] '))
fps = fps_multiplier * 30
output_video_path = 'out2.mp4'
frames = []

# get all files in the directory
file_names = [f for f in os.listdir(temp_directory) if isfile(join(temp_directory, f))]

# sort frame file names
file_names.sort(key = lambda x: int(x[:-4]))

for i in range(len(file_names)):
	filename = temp_directory + '\\' + file_names[i]
	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width, height)
	frames.append(img)

# set frames per second(fps) and size and write into video frame by frame
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frames)):
	out.write(frames[i])
out.release()

# delete temporary directory
shutil.rmtree(temp_directory)

print('Successfully Converted Video %s to Video %s with %f times more fps' % (video_path, output_video_path, fps_multiplier))
