from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed="normal") #скорость - “normal”, “fast”, “faster”, “fastest” и “flash”

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "test_video.mp4"),
                            output_file_path=os.path.join(execution_path, "detected")
                            , frames_per_second=30, log_progress=True)
print(video_path)
