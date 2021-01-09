import cv2
import numpy as np

def pickup(img):
    img = cv2.resize(img,(960,480))
    imgCropped = img[:470,240:710]
    imgBlurred = cv2.GaussianBlur(imgCropped,(5,5),0)
    imgGray = cv2.cvtColor(imgBlurred,cv2.COLOR_BGR2GRAY)

    th1 = cv2.getTrackbarPos("CannyTh1","Parameters")
    th2 = cv2.getTrackbarPos("CannyTh2","Parameters")
    areaMin = cv2.getTrackbarPos("minArea","Parameters")
    areaMax = cv2.getTrackbarPos("maxArea","Parameters")

    imgCanny = cv2.Canny(imgGray,th1,th2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    (cnts,_) = cv2.findContours(imgDil.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    objects = imgCropped.copy()
    objCount = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area>areaMin and area<areaMax:
            objCount += 1
            # cv2.drawContours(objects,cnt,-1,(0,255,255),3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            # cv2.rectangle(objects,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.circle(objects,(x+(w//2),y+(h//2)),5,(0,255,0),-1)
            rect = cv2.minAreaRect(cnt)     #For rotated rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(objects,[box],0, (0,0,255),2)

    
    text= f"No of objects detected=  {objCount}"
    cv2.putText(objects,text,(60,460),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    # cv2.imshow('inputimage', imgCropped)
    # cv2.imshow('edgedimage', imgCanny)
    cv2.imshow('outputimage', objects)
    return objects


def dropto(imgD):
    imgD = cv2.resize(imgD,(960,480))
    imgCroppedD = imgD[10:465,250:705]
    imgBlurredD = cv2.GaussianBlur(imgCroppedD,(5,5),0)
    imgGrayD = cv2.cvtColor(imgBlurredD,cv2.COLOR_BGR2GRAY)


    th1 = cv2.getTrackbarPos("CannyTh1","Parameters")
    th2 = cv2.getTrackbarPos("CannyTh2","Parameters")
    areaMin = cv2.getTrackbarPos("minArea","Parameters")
    areaMax = cv2.getTrackbarPos("maxArea","Parameters")
    

    imgCannyD = cv2.Canny(imgGrayD,th1,th2)
    kernelD = np.ones((5, 5))
    imgDilD = cv2.dilate(imgCannyD, kernelD, iterations=1)
    (cntsD,_) = cv2.findContours(imgDilD.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    objectsD = imgCroppedD.copy()
    objCountD = 0
    for cntD in cntsD:
        areaD = cv2.contourArea(cntD)
        if areaD>areaMin and areaD<areaMax:
            objCountD += 1
            # cv2.drawContours(objects,cnt,-1,(0,255,255),3)
            peri = cv2.arcLength(cntD, True)
            approx = cv2.approxPolyDP(cntD, 0.02 * peri, True)
            # print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            # cv2.rectangle(objects,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.circle(objectsD,(x+(w//2),y+(h//2)),5,(0,255,0),-1)
            rect = cv2.minAreaRect(cntD)     #For rotated rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(objectsD,[box],0, (0,0,255),2)
    text= f"No of grids=  {objCountD}"
    cv2.putText(objectsD,text,(100,420),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    # cv2.imshow('inputimageD', imgCroppedD)
    # cv2.imshow('edgedimageD', imgCannyD)
    cv2.imshow('outputimageD', objectsD)
    return objectsD


def scale_contour(cnt,scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx,cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,200)
cv2.createTrackbar("CannyTh1","Parameters",17,255,empty)
cv2.createTrackbar("CannyTh2","Parameters",125,255,empty)
cv2.createTrackbar("minArea","Parameters",1000,30000,empty)
cv2.createTrackbar("maxArea","Parameters",3000,350000,empty)


pickFiles = ['areavideos/pick1.mp4','areavideos/pick2.mp4','areavideos/pick3.mp4']
dropFiles = ['areavideos/drop1.mp4','areavideos/drop2.mp4','areavideos/drop3.mp4']
for i in range(3):
    pickcamplay=cv2.VideoCapture(pickFiles[i])
    dropcamplay=cv2.VideoCapture(dropFiles[i])
    while(pickcamplay.isOpened() or dropcamplay.isOpened()):
        check, img = pickcamplay.read()
        checkD, imgD = dropcamplay.read()
        if (check==False and checkD==False):
            break

        elif(checkD == False):
            pickup(img)

        elif(check == False):
            dropto(imgD)
            
        else:
            pickup(img)
            dropto(imgD)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    pickcamplay.release()
    dropcamplay.release()
    
cv2.destroyAllWindows()