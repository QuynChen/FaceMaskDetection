from detector import Detector
import cv2
import streamlit as st
from PIL import Image
import numpy as np


def file():
    inputimg = st.file_uploader("Upload your image")
    if inputimg is not None:
        inputimg = Image.open(inputimg)
        inputimg = np.array(inputimg)
        inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('input.jpg', inputimg)
        return inputimg
    
def webcam():
    inputimg = st.camera_input("Take a picture")
    if inputimg is not None:
        inputimg = Image.open(inputimg)
        inputimg = np.array(inputimg)
        inputimg = cv2.cvtColor(inputimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('input.jpg', inputimg)
        return inputimg
    
def statistics(counter, time):
    total = float(sum(counter.values()))
    print(total)
    print(counter)
    withoutm = float(counter[1])
    withm = float(counter[0])
    inc = float(counter[2])
    st.write("Tổng số người có trong bức ảnh: ", total)
    st.write("Tổng số người không đeo khẩu trang: ", withoutm)
    st.write("Tổng số người đeo khẩu trang sai cách: ", inc)
    st.write("Tổng số người đeo khẩu trang: ", withm)
    st.write("Tỉ lệ số người không đeo khẩu trang: ", round(withoutm/total, 2))
    st.write("Tỉ lệ số người đeo khẩu trang sai cách: ", round(inc/total, 2))
    st.write("Tỉ lệ số người đeo khẩu trang: ", round(withm/total, 2))
    st.write("Thời gian thực thi (miliseconds): ", time)

config_files = ['faster_rcnn', 'yolof']

faster_rcnn_model = Detector(f'configs/{config_files[0]}.py', f'weights/{config_files[0]}.pth')
yolof_model = Detector(f'configs/{config_files[1]}.py', f'weights/{config_files[1]}.pth')

st.title("Demo đồ án môn học CS338 - Nhận dạng")

st.write("Trần Thị Mỹ Quyên - 20520731")
st.write("Nguyễn Thị Thùy - 21521514")
st.write("Phan Thành Phúc - 21522477")

file_page, webcam_page = st.tabs(["File", "Webcam"])

with file_page:
    inputimg_file = file()
    if inputimg_file is not None:
        st.image(cv2.cvtColor(inputimg_file, cv2.COLOR_BGR2RGB))
        faster_rcnn, yolof = st.columns(2)
        with faster_rcnn:
            st.write('Faster-RCNN')
            result_faster_rcnn, counter_faster_rcnn, faster_rcnn_time = faster_rcnn_model.detect('input.jpg')
            
            st.image(cv2.cvtColor(result_faster_rcnn, cv2.COLOR_BGR2RGB))
            statistics(counter_faster_rcnn, faster_rcnn_time)
        with yolof:
            st.write('YOLOF')
            result_yolof, counter_yolof, yolof_time = yolof_model.detect('input.jpg')
            
            st.image(cv2.cvtColor(result_yolof, cv2.COLOR_BGR2RGB))
            statistics(counter_yolof, yolof_time)

with webcam_page:
    inputimg_wc = webcam()
    if inputimg_wc is not None:
        st.image(cv2.cvtColor(inputimg_wc, cv2.COLOR_BGR2RGB))
        faster_rcnn, yolof = st.columns(2)
        with faster_rcnn:
            st.write('Faster-RCNN')
            result_faster_rcnn, counter_faster_rcnn, faster_rcnn_time = faster_rcnn_model.detect('input.jpg')
            
            st.image(cv2.cvtColor(result_faster_rcnn, cv2.COLOR_BGR2RGB))
            statistics(counter_faster_rcnn, faster_rcnn_time)
        with yolof:
            st.write('YOLOF')
            result_yolof, counter_yolof, yolof_time = yolof_model.detect('input.jpg')
            
            st.image(cv2.cvtColor(result_yolof, cv2.COLOR_BGR2RGB))
            statistics(counter_yolof, yolof_time)

