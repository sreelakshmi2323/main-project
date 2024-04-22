import glob
import cv2
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
NUM_CLUSTERS = 5
cv_img = []
cnt=0
for img in glob.glob("dataset/dark/*.jpg"):
    print('reading image')
    im = Image.open(img)
    im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    print('most frequent is %s (#%s)' % (peak, colour))
    ccode="#"+str(colour)
    from PIL import Image
    
    # color --> "red" or (255,0,0) or #ff0000
    img = Image.new('RGB',(200,200),ccode)
    im1 = img.save('colordata/dark/dark'+str(cnt)+".jpg")
    cnt+=1
    print(cnt)