"""
Video to frame conversion with multiprocessing / 使用多处理的视频到帧转换
This module converts videos to frames using multiprocessing and OpenCV for efficient processing.
此模块使用多处理和OpenCV将视频转换为帧，以实现高效处理。

Dataset structures / 数据集结构:
Shanghaitech:
|------------|videos:
|------------------|Train:
|------------------------|: video1
|------------------------|: video2
|------------------------|: video3
|------------------------|: video4 ...

UCF_Crime:
|------------|videos:
|------------------|Class1:
|------------------------|: video1
|------------------------|: video2
|------------------------|: video3
|------------------------|: video4 ...
|------------------|Class2:

Frame structures / 帧结构:
Shanghaitech:
|------------|frames:
|------------------|Train:
|------------------------|: video1
|------------------------|: video2
|------------------------|: video3
|------------------------|: video4 ...

UCF_Crime:
|------------|frames:
|------------------|Class1:
|------------------------|: video1
|------------------------|: video2
|------------------------|: video3
|------------------------|: video4 ...
|------------------|Class2:
"""


from multiprocessing import Pool
import os
import platform
import cv2
import glob
import time
from tqdm import tqdm


def readVideolist2(Train,Test):
    """
    For Avenue, ped1, ped2
    :param Train:
    :param Test:
    :return:
    """
    Trainvideos = os.listdir(Train)
    Testvideos = os.listdir(Test)
    TrainvideoList={}
    TestvideoList={}
    for video in Trainvideos:
        TrainvideoPath = os.path.join(Train, video)
        TrainvideoList[video] = TrainvideoPath
    for video in Testvideos:
        TestvideoPath = os.path.join(Test, video)
        TestvideoList[video] = TestvideoPath
    return TrainvideoList, TestvideoList

def readVideolist3(Train):
    """
    For shanghaitech
    :param Train:
    :param Test:
    :return:
    """
    Trainvideos = os.listdir(Train)
    TrainvideoList={}
    for video in Trainvideos:
        TrainvideoPath = os.path.join(Train, video)
        TrainvideoList[video] = TrainvideoPath
    return TrainvideoList

def readVideolist_UCF_Crime(Path):
    """
    For UCF_Crime
    :param Train:
    :param Test:
    :return:
    """
    Class = os.listdir(Path)
    videoList={}
    for a_class in Class:
        videoDir=os.path.join(Path, a_class)
        videoNames=os.listdir(videoDir)
        for video in videoNames:
            videoPath=os.path.join(videoDir,video)
            videoList[a_class+video]=videoPath
    return videoList


def Video2frame(videofilelist=None):
    # for each_video in videofilelist:
        each_video = videofilelist
        if each_video.rfind('UCF') != -1:
            zfill_number = 6
        else:
            zfill_number = 5
        current_os = platform.architecture()
        start = time.clock()
        # print(videofilelist)
        if current_os[1] == 'WindowsPE':
            each_videolist = each_video.split('\\')
            each_video_path, each_video_name = os.path.split(each_video)
        else:
            each_videolist = each_video.split('/')
            each_video_path, each_video_name = os.path.split(each_video)
        each_video_name, _ = each_video_name.split('.')
        each_video_name = str(each_video_name)

        target_path = os.path.join(each_video_path.replace('Videos','frames'),each_video_name)
        # target_path = target_path.replace('windows4t','windowswan')
        if os.path.exists(target_path):
            print('Frames of Video_{} already existed'.format(each_video_name))
        else:
            framedirsavepath = target_path
            if os.path.exists(framedirsavepath) == 0:
                os.makedirs(framedirsavepath)
            cap = cv2.VideoCapture(each_video)
            frame_count = 1
            success = True
            while(success):
                success, frame = cap.read()
                # print ('Read a new frame: ', success)
                if success:
                    cv2.imwrite(framedirsavepath + '/'+"img_{}.jpg" .format(str(frame_count).zfill(zfill_number)), frame)
                    # print ('frame_num:',frame_count)
                    frame_count = frame_count + 1
                else:
                    print('Preparing for next video')
                    end = time.clock()
                    t = end - start
                    print("Consuming {} seconds for video {}：".format(t, each_video_name))
                    break
            cap.release()


if __name__ == "__main__":
    current_os = platform.architecture()
    # TrainSetpath2='/home/tu-wan/windowswan/dataset/Avenue/Videos/training_videos/'
    # TestSetPath2 ='/home/tu-wan/windowswan/dataset/Avenue/Videos/testing_videos/'

    # UCFSetpath='/home/tu-wan/windowswan/dataset/UCF_Crime/Videos/'
    TrainSetpath='/home/tu-wan/windowswan/dataset/LV/Videos'

    #######It should notices that the readVideolist* should be selected for dataset.
    TrainvideoList = readVideolist3(TrainSetpath)
    train_values = list(TrainvideoList.values())

    # TrainFrameList, TestFrameList = readVideolist(TrainFramepath, TestFramePath)
    # train_values = list(TrainvideoList.values())

    with Pool(processes=4) as p:
        max_ = len(train_values)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(Video2frame, train_values))):
                pbar.update()
