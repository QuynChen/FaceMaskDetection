<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>NHẬN DẠNG</b></h>

## THÀNH VIÊN NHÓM

| STT | MSSV        | Họ và Tên                 | Github                                             | Email                      |
| :-: | :---------: | :-----------------------: | :------------------------------------------------: |:--------------------------:|
| 1   | 20520731    | Trần Thị Mỹ Quyên         |[QuynChen](https://github.com/QuynChen)             | ttmyquyen.ns2002@gmail.com |
| 2   | 21521514    | Nguyễn Thị Thùy           |[thuynguyen2003](https://github.com/thuynguyen2003) |thuynguyen28102003@gmail.com|
| 3   | 21522477    | Phan Thành Phúc           |[tumble29](https://github.com/tumble29)             |21522477@gm.uit.edu.vn      |

## GIỚI THIỆU MÔN HỌC

-   **Tên môn học:** Nhận Dạng 
-   **Mã môn học:** CS338
-   **Mã lớp:** CS338.N21
-   **Năm học:** HK2 (2022 - 2023)
-   **Giảng viên**: Đỗ Văn Tiến 

## ĐỒ ÁN CUỐI KÌ

**Đề tài**: FACE MASK DETECTION

### **I. Giới thiệu**

Phát hiện khẩu trang là bài toán thuộc nhóm các bài toán phát hiện đối tượng của lĩnh vực thị giác máy tính và có nhiều ứng dụng thực tế. Mục tiêu của bài toán là xác định xem một người có đang đeo khẩu trang hay không dựa trên ảnh hoặc video chứa hình ảnh của họ. Cụ thể, có thể phát biểu rằng:
   * **Đầu vào (input)**: Hình ảnh người đeo hoặc không đeo khẩu trang được trích xuất từ webcam hoặc camera.
   * **Đầu ra (output)**: Các thông tin cơ bản tương ứng với từng người có trong ảnh đầu vào: vị trí (bounding-box bao quanh khuôn mặt người), label mà người đó thuộc về.
<div align="center">
    <img width="500" alt="Input và Output bài toán phát hiện khẩu trang" src="https://github.com/QuynChen/FaceMaskDetection/assets/127386286/a898d1d5-c8c5-44f1-87dd-889a87ed3358">
</div>


### **II. Bộ dữ liệu** 

Tải bộ dữ liệu [tại đây](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

Bộ dữ liệu gồm 853 ảnh, có định dạng PNG với đa dạng kích thước hình ảnh. Mỗi hình ảnh trong bộ dữ liệu đi kèm với một file XML tương ứng. File XML chứa các trường thông tin như tọa độ bbox (x,y,width,height) của khuôn mặt và nhãn của các đối tượng trong ảnh.
<div align="center">
  <img width="500" alt="Hình ảnh trong bộ dữ liệu" src="https://github.com/QuynChen/FaceMaskDetection/assets/127386286/f83dd2fe-7791-4591-88a5-da617166e7e8">
  <p align="center"> Một số hình ảnh trong bộ dữ liệu Face Mask Detection </p>
</div>

**Thông tin dữ liệu thực nghiệm:**
1) Số lượng:
-  Train set: 545 ảnh
-  Validation set: 137 ảnh
-  Test set: 171 ảnh
2) Labels:
-  *with_mask*: Người đeo khẩu trang đúng cách, nghĩa là che kín mũi và miệng.
-  *without_mask*: Người không đeo khẩu trang.
-   *mask_weared_incorrect*: Người đeo khẩu trang không đúng cách, nghĩa là không che kín mũi, không che kín miệng hoặc không che kín cả mũi và miệng
<p align="center">
  <img width="500" alt="Labels trong bộ dữ liệu" src="https://github.com/QuynChen/FaceMaskDetection/assets/127386286/c31ed36e-75fe-43d2-9352-64d7ddfea896">
</p>


### **III. Kết quả and Models**

**Kết quả đánh giá mô hình dựa trên độ đo mAP của các phương pháp thực nghiệm**

| Method       | Backbone  | Style     | Epoch | Lr      | box AP   |   Config      | 
| :----------: | :-------: | :-------: | :---: | :-----: | :------: | :-----------: | 
| Faster RCNN  | ResNet 50 | Pytorch   |   40  |   0.01  | 0.508    | [config](https://github.com/QuynChen/FaceMaskDetection/blob/main/config/faster-rcnn_r50_fpn_1x_coco.py)    |
| YOLOv3       | Darknet 53| Pytorch   |   273 |   0.001 | 0.369    | [config](https://github.com/QuynChen/FaceMaskDetection/blob/main/config/yolov3_d53_8xb8-320-273e_coco.py)    |
| YOLOv5       | CSPDarknet| Pytorch   |   200 |   0.01  | 0.57     | [config](https://github.com/QuynChen/FaceMaskDetection/blob/main/config/yolov5s.yaml)    |
| YOLOF        | ResNet 50 | Caffe     |   32  |   0.01  | 0.445    | [config](https://github.com/QuynChen/FaceMaskDetection/blob/main/config/yolof_r50-c5_8xb8-1x_coco.py)    |

**Kết quả đánh giá dựa trên độ đo mAP của các class:**

| Method       | Backbone  |  with_mask   |   without_mask |  mask_weared_incorrect |
| :----------: | :-------: | :----------: | :------------: | :--------------------: | 
| Faster RCNN  | ResNet 50 |   0.653      |	0.493	         | 0.378                  |
| YOLOv3       | Darknet 53|   0.535	    | 0.279	         | 0.293                  | 
| YOLOv5       | CSPDarknet|  0.708	      | 0.536	         | 0.467                  |
| YOLOF        | ResNet 50 |  0.544	      | 0.389	         | 0.404                  |   

## DEMO

Để chạy demo streamlit, trước tiên cần tải file weight [Faster-RCNN](https://github.com/QuynChen/FaceMaskDetection/releases/download/FaceMaskDetectionModel/faster_rcnn.pth), [YOLOF](https://github.com/QuynChen/FaceMaskDetection/releases/download/FaceMaskDetectionModel/yolof.pth). Sau đó đặt các file weight vừa tải vào folder model trong DemoRecognize.

### Cài đặt môi trường

**Bước 1:** Tạo môi trường conda và active nó

```
cd DemoRecognize
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Bước 2:** Cài Pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Bước 3:** Cài MMEngine và MMCV bằng MIM.

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Bước 4:** Cài MMDetection.

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

**Bước 5:** Cài đặt Streamlit và OpenCV.

```
conda install -c conda-forge streamlit
conda install -c conda-forge opencv
```

**Bước 6:** Cuối cùng, chạy dòng lệnh đơn giản sau.

```
streamlit run demo.py
```

## Citation

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
