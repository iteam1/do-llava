'''
python3 cite_images.py labels/chat.json
'''
import os
import sys
import json
import random
import shutil

# Init
f = 0.005
destintion_path = 'dst'
json_path = sys.argv[1]
 
# Opening JSON file
with open(json_path) as json_file:
    data = json.load(json_file)

# Create destination dir
if not os.path.exists(destintion_path):
    os.mkdir(destintion_path)

N = len(data)
n = int(f * N)
print(f'cite {f} {n}/{N}')
samples = random.choices(data, k=n)

for i,row in enumerate(samples):
    image=row['image']
    src = os.path.join('images',image)
    dst = os.path.join(destintion_path,image)
    shutil.copy(src,dst)
    print(i,src,os.path.isfile(src))

# Serializing json
json_object = json.dumps(samples, indent=4)
 
# Writing to sample.json
with open("dst.json", "w") as outfile:
    outfile.write(json_object)
