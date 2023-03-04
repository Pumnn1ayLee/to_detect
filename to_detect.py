import numpy as np
import numpy
import cv2
import dlib
import os
import glob

# 人脸关键点检测器路径
predictor_path = "E://Download/shape_predictor_68_face_landmarks.dat.dat"
# 人脸识别模型路径
face_rec_model_path = "E://Download/dlib_face_recognition_resnet_model_v1.dat.dat"
# 候选人脸图片路径
faces_folder_path = "D://pythonProject5/candidate1/"

# 1、创建正脸检测器detector
detector = dlib.get_frontal_face_detector()
# 2、创建人脸关键点预测器sp
sp = dlib.shape_predictor(predictor_path)
# 3、创建人脸识别器facerec
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
# 创建候选人脸描述子列表descriptors,初始状态为空
descriptors = []
# 候选人名单
candidate = ['Lifurong''Linzhenhua']
# 对HX文件夹下的每一个候选人脸进行:
# 1.人脸检测
# 2.关键点检测
# 3.描述子提取
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print(f)
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 1.人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # 2.关键点检测
        shape = sp(img, d)
        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 转换为numpy array
        v = numpy.array(face_descriptor)
        descriptors.append(v)
numpy.save('descriptor1.npy', descriptors)
print(len(descriptors))