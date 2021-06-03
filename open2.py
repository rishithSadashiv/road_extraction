from water import watershed,sharpen,auto_canny,blobdetect
import numpy as np
import cv2

frame2 = cv2.imread('dataset/img3.jpg') #img load local_dataset
# frame2 = cv2.imread('local_dataset/apmc_yard.png')
# frame2 = cv2.imread('local_dataset/batawadi_higway.png')
# frame2 = cv2.imread('local_dataset/hemavathi.png')
# frame2 = cv2.imread('local_dataset/hemavathi_river.png')
# frame2 = cv2.imread('local_dataset/hemavathi2.png')
# frame2 = cv2.imread('local_dataset/hemavathi3.png')
# frame2 = cv2.imread('local_dataset/kaver5.png')
# frame2 = cv2.imread('local_dataset/kaveri.png')
# frame2 = cv2.imread('local_dataset/kaveri2.png')
# frame2 = cv2.imread('local_dataset/kaveri3.png')
# frame2 = cv2.imread('local_dataset/kaveri4.png')
# frame2 = cv2.imread('local_dataset/kaveri6.png')
# frame2 = cv2.imread('local_dataset/shivkscircle.png')
# frame2 = cv2.imread('local_dataset/shivkumar_swamiji_circle.png')
# frame2 = cv2.imread('local_dataset/shivkumarswamijicircle_bhuvan.png')
# frame2 = cv2.imread('local_dataset/tumkur.png')
# frame2 = cv2.imread('local_dataset/tumkur2.png')

sharped=sharpen(frame2)

frame=cv2.medianBlur(frame2,5)
#black=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

Z = frame.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
# print(ret)
resd = center[label.flatten()]
resd2 = resd.reshape((frame.shape))
cluster=resd2
resd2=cv2.cvtColor(resd2,cv2.COLOR_BGR2GRAY)
#retval, resd2 = cv2.threshold(resd2, 150, 255, cv2.THRESH_BINARY)


#resd2=cv2.medianBlur(resd2,5)
sharpcluster=sharpen(resd2)

##########################################canny edge####################################################
edges = cv2.Canny(frame,400,600)
#edges2=cv2.Canny(img_dilation,249,250)
edges3=cv2.Canny(resd,200,260)
edgesharped=auto_canny(sharpcluster)
###########################################smoothen kmeans output##################################################
smoothmeans=cv2.GaussianBlur(resd2,(5,5),0)
##########################################################histogram equalisation##########################################
equ = cv2.equalizeHist(resd2)

####################################BLOBDETECT######################
final=blobdetect(equ)




# cv2.imshow("input image",frame2)
# cv2.imshow("GaussianBlur output",frame)
# cv2.imshow("clustering",cluster)
# cv2.imshow("histogram",equ)
# cv2.imshow("contour output",final)
# final.shape
# cv2.imshow("grey conversion",resd2)
retval, threshold = cv2.threshold(final, 220, 255, cv2.THRESH_BINARY)
cv2.imshow("final",threshold)
#watershed(threshold)

img2 = np.zeros_like(frame2)
img2[:,:,2] = threshold
added_image = cv2.addWeighted(frame2,0.9,img2,0.9,0)
cv2.imshow("new", img2)
cv2.imshow("input image",frame2)
cv2.imshow("GaussianBlur output",frame)
cv2.imshow("clustering",cluster)
cv2.imshow("histogram",equ)
cv2.imshow("contour output",final)
cv2.imshow("roads recognised", added_image)

#.imshow('log',laplacian)
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()