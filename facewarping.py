import numpy as np 
import cv2
import sys
import dlib
import functools
import imutils
from imutils import face_utils
import math


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#image = cv2.imread("averagefaceimage_setb.jpg")
#im_copy = image.copy()
#image = imutils.resize(image, width=500)

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True    

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform

def read_landmarks(image):

    rects = cascade.detectMultiScale(image)

    x,y,w,h = rects[0].astype(int)
    rect = dlib.rectangle(x, y, x + w, y + h)

    face_points = predictor(image,rect).parts()
    #face_points = face_utils.shape_to_np(face_points)
    #return face_points
    landmarks = []
    for n in face_points:
        landmarks.append([n.x, n.y])
    return landmarks
    

#only used for visualizing delauney triangles
def triDistanceSort(a, b):
    #we want to sort by distance to the POI (top left corner)
    x=0
    y=0
    #find the center point of triangle a
    aX = (a[0]+a[2]+a[4]) / 3;
    aY = (a[1]+a[3]+a[5]) / 3;
    #find the center point of triangle b
    bX = (b[0] + b[2] + b[4]) / 3;
    bY = (b[1] + b[3] + b[5]) / 3;
    #find distance from triangle a to the POI
    distA=np.sqrt( (aX - x)**2 + (aY - y)**2 )
    #find distance from triangle b to the POI
    distB = np.sqrt((bX - x) ** 2 + (bY - y) ** 2)

    #return the conventional comparator outputs
    if distA > distB:
        return -1
    elif distA < distB:
        return 1
    else:
        return 0

#function for visualising the triangle
def drawColoredTriangles(img, triangleList, disp):
    #sort the triangle list by distance from the top left corner in order to get a gradient effect when drawing triangles
    triangleList=sorted(triangleList, key=functools.cmp_to_key(triDistanceSort))
    h, w, c= img.shape
    #get bounding rectangle points of image
    r = (0, 0, w, h)
    #iterate through and draw all triangles in the list
    for idx, t in enumerate(triangleList):
        #grab individual vertex points
        pt1 = [t[0], t[1]]
        pt2 = [t[2], t[3]]
        pt3 = [t[4], t[5]]
        #select a position for displaying the enumerated triangle value
        pos = (t[2], t[3])
        #create the triangle
        triangle = np.array([pt1, pt2, pt3], np.int32)
        #select a color in HSV!! (manipulate idx for cool color gradients)
        #color = np.uint8([[[idx, 100, 200]]])
        color = np.uint8([[[0, 0, idx]]])
        #convert color to BGR
        bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        color = (int(bgr_color[(0, 0, 0)]), int(bgr_color[(0, 0, 1)]), int(bgr_color[(0, 0, 2)]))

        #draw the triangle if it is within the image bounds
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.fillPoly(img, [triangle], color)
            # if display triangle number was selected, display the number.. this helps with triangle manipulation later
            if(disp==1):
                cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(0, 0, 0))

def readLandmarks(image) :
    image = cv2.imread(image)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 1)
    for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	    shape = predictor(image, rect)
	    shape = face_utils.shape_to_np(shape)
    return shape

def gettingpointsOffset(image):
    fp = readLandmarks(image)
    facepoints = []
    n = 0
    while n<68:
        x = fp[n,0]
        y = fp[n,1]
        facepoints.append((int(x),int(y)))
        n += 1

    image1 = cv2.imread(image)
    #image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image1 = np.float32(image1)/255.0
    w= 250
    h = 250
    eyecornerdst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]
    eyecornersrc = [facepoints[36], facepoints[45]]

    tform = similarityTransform(eyecornersrc, eyecornerdst)    
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])

    afterImage = cv2.warpAffine(image1, tform, (w,h))

    points1 = np.reshape(np.array(facepoints), (68,1, 2))

    points2 = cv2.transform(points1, tform)

    points2 = np.float32(np.reshape(points2, (68, 2)))

    return points2

OffsetAVG = gettingpointsOffset('averagefaceimage_seta.jpg')
OffsetSmile = gettingpointsOffset('averagefaceimage_setb.jpg')

SmileoffsetPoints = OffsetSmile - OffsetAVG
SmileoffsetPoints = np.asarray(SmileoffsetPoints)
#print(SmileoffsetPoints)
#print(len(SmileoffsetPoints))

image = cv2.imread("1b.jpg")
#image = imutils.resize(image, width=250)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_copy = image.copy()
h = 300
w = 250
c = 3
rect = (0,0,w,h)

subdiv = cv2.Subdiv2D(rect)
points = read_landmarks(image)
#points = np.float32(np.reshape(points, (68, 2)))
points = np.asarray(points)
print(points)
#print(len(points))
print(SmileoffsetPoints)
pointsforwarp = points + SmileoffsetPoints
#pointsforwarp = np.asarray(pointsforwarp)
#pointsforwarp[0,0] = 0
print(pointsforwarp)

for p in points:
    subdiv.insert((p[0],p[1]))
#adding boundaries for triangulation

subdiv.insert((0, h - 1))
subdiv.insert((w / 2, h - 1))
subdiv.insert((w / 2, 0))
subdiv.insert((0, h / 2))
subdiv.insert((w - 1, h / 2))
subdiv.insert((w - 1, h - 1))
subdiv.insert((w - 1, 0))
subdiv.insert((1, 1))


smileAVG = cv2.imread("1b.jpg")
smileCPY = smileAVG.copy()

subdiv2 = cv2.Subdiv2D(rect)

for i in pointsforwarp:
    subdiv2.insert((i[0],i[1]))
#adding boundaries for triangulation

subdiv2.insert((0, h - 1))
subdiv2.insert((w / 2, h - 1))
subdiv2.insert((w / 2, 0))
subdiv2.insert((0, h / 2))
subdiv2.insert((w - 1, h / 2))
subdiv2.insert((w - 1, h - 1))
subdiv2.insert((w - 1, 0))
subdiv2.insert((1, 1))


#print
#drawColoredTriangles(image, subdiv.getTriangleList(), 0)
#cv2.imshow('triangles vis', image)

#blended =  cv2.addWeighted(im_copy, 0.3, image, 0.7, 0)
#cv2.imshow('back/front', blended)

#cv2.waitKey(0)
#cv2.destroyAllWindows()



#######################################################################

smilingImage = 255 * np.ones(image.shape, dtype=image.dtype)
#np.zeros((h,w,3), np.float32())

originalTriList = subdiv.getTriangleList()

#destinationlst is a copy of original to warp
#destinationLst = originalTriList.copy()

#technically, all you need is the smiling triangle list

smilingLst = subdiv2.getTriangleList()

#print(len(smilingLst))
#print(len(originalTriList))
#print(smilingLst[0, 0])
#print(originalTriList[0, 0])
#drawColoredTriangles(im_copy, smilingLst,0)
    

for src, dst in zip(originalTriList, smilingLst):
    pt1 = [src[0], src[1]]
    pt2 = [src[2], src[3]]
    pt3 = [src[4], src[5]]

    if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
        tri1 = np.float32([[[src[0], src[1]], [src[2], src[3]], [src[4], src[5]]]])
        #print(tri1)
        tri2 = np.float32([[[dst[0], dst[1]], [dst[2], dst[3]], [dst[4], dst[5]]]])
        #print(tri2)
        r1 = cv2.boundingRect(np.float32(tri1))
        r2 = cv2.boundingRect(np.float32(tri2))
        tri1Cropped = []
        tri2Cropped = []
        tri2CroppedINT = []

        for i in range(0, 3):
            tri1Cropped.append(((tri1[0][i][0] - r1[0]), (tri1[0][i][1] - r1[1])))
            tri2Cropped.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))
            tri2CroppedINT.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))

        M = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
        img1Cropped = im_copy[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        warpedImageCropped = cv2.warpAffine(img1Cropped, M, (r2[2], r2[3]), None,  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri2CroppedINT), (1.0, 1.0, 1.0), 16, 0)
        # Apply mask to cropped region

        #print(r2)
        #print(warpedImageCropped.shape)
        #print(mask.shape)
        warpedImageCropped = warpedImageCropped * mask
        #print(warpedImageCropped.shape)

        # Copy triangular region of the rectangular patch to the output image
        smilingImage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = smilingImage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
        smilingImage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = smilingImage[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + warpedImageCropped

cv2.imshow('Smiling NOW!', smilingImage)
cv2.imshow('ORIGNAL?', im_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()