import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# تحميل نموذج YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# إعداد واجهة Streamlit
st.set_page_config(page_title="Real-Time Pose Estimation", layout="wide")
st.title("👁️ كشف مفاصل الجسم الحي - YOLOv8 Pose")
st.markdown("---")

# اختيار طريقة الإدخال
option = st.radio("اختر طريقة الإدخال:", ["📷 الكاميرا المباشرة", "📁 رفع صورة"], horizontal=True)

# دالة لرسم المفاصل المطلوبة فقط
def draw_custom_joints(frame, keypoints, conf_threshold=0.5):
    # المفاصل المطلوبة فقط
    JOINT_INDICES = {
        'shoulder': [5, 6], 
        'elbow': [7, 8], 
        'wrist': [9, 10],
        'hip': [11, 12], 
        'knee': [13, 14]
    }

    COLORS = {
        'shoulder': (0, 255, 0),    # أخضر
        'elbow': (255, 0, 0),       # أزرق
        'wrist': (0, 0, 255),       # أحمر
        'hip': (255, 255, 0),       # سماوي
        'knee': (0, 255, 255)       # أصفر
    }

    for joint_type, indices in JOINT_INDICES.items():
        for idx in indices:
            kp = keypoints[idx]
            if kp[2] >= conf_threshold:  # الثقة
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 6, COLORS[joint_type], -1)

    # خطوط الربط بين المفاصل
    connections = [(5,7), (6,8), (7,9), (8,10), (11,13), (12,14)]  
    for (start, end) in connections:
        kp_start, kp_end = keypoints[start], keypoints[end]
        if kp_start[2] >= conf_threshold and kp_end[2] >= conf_threshold:
            start_pt = (int(kp_start[0]), int(kp_start[1]))
            end_pt = (int(kp_end[0]), int(kp_end[1]))
            cv2.line(frame, start_pt, end_pt, (255, 255, 255), 2)

    return frame

# معالجة الإطار باستخدام YOLO
def process_frame(frame):
    results = model(frame)
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # استخراج النقاط
        return draw_custom_joints(frame, keypoints)
    return frame

# تشغيل الكاميرا المباشرة
if option == "📷 الكاميرا المباشرة":
    live_cam = st.checkbox("✅ تشغيل الكاميرا")
    
    if live_cam:
        cap = cv2.VideoCapture(0)  # تشغيل الكاميرا
        
        if not cap.isOpened():
            st.error("❌ لم يتم العثور على كاميرا!")
        else:
            stframe = st.empty()
            while live_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ تعذر تشغيل الكاميرا!")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = process_frame(frame)
                stframe.image(processed, use_column_width=True)
            
            cap.release()

# رفع صورة ومعالجتها
elif option == "📁 رفع صورة":
    uploaded_file = st.file_uploader("🖼️ اختر صورة", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = process_frame(frame)
        st.image(processed, caption="🔍 الكشف عن المفاصل", use_column_width=True)
