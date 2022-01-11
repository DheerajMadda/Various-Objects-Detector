import os
import cv2
import numpy as np
import torch
import tempfile
import streamlit as st


ckpt_path = os.path.join(os.getcwd(), 'yolov5s.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt_path)


if __name__ == "__main__":

    st.title("Various Objects Detector")
    st.subheader("Home")
    VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm"]
    video_file = st.file_uploader("Upload the video", type=VIDEO_EXTENSIONS)

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            img = np.squeeze(results.render())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img)
