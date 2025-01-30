# تثبيت المكتبات المطلوبة
!pip install ultralytics streamlit opencv-python-headless pyngrok streamlit-webrtc -q  

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import tempfile
import os
from pyngrok import ngrok
from subprocess import Popen

# 🔐 إعداد ngrok باستخدام authtoken (استبدله بـ التوكن الخاص بك)
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTHTOKEN"  # ⛔ استبدل التوكن هنا 🔴
!ngrok authtoken {NGROK_AUTH_TOKEN}

# تحميل نموذج YOLOv8-Pose
model = YOLO('yolov8n-pose.pt')

# 📌 إنشاء تطبيق Streamlit
with open("app.py", "w") as f:
    f.write("""
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

# تحميل النموذج
model = YOLO('yolov8n-pose.pt')

st.set_page_config(page_title="YOLOv8 Pose Estimation", layout="wide")
st.title('👁️ كشف مفاصل الجسم باستخدام YOLOv8 Pose')
st.markdown("---")

st.write("✅ التطبيق جاهز! اختر طريقة الإدخال.")

option = st.radio("اختر الإدخال:", ["📷 الكاميرا", "📁 رفع صورة"], horizontal=True)

def process_frame(frame):
    results = model.predict(frame, conf=0.6, imgsz=320)
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints[0].cpu().numpy()
        for i in range(17):
            x, y, conf = keypoints[i]
            if conf >= 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    return frame

if option == "📁 رفع صورة":
    uploaded_file = st.file_uploader("📂 ارفع صورة", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed = process_frame(frame)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="🔍 النتيجة", use_column_width=True)

elif option == "📷 الكاميرا":
    st.write("📸 استخدم الكاميرا المباشرة من متصفحك:")
    
    def video_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processed = process_frame(img)
        return processed
    
    webrtc_streamer(key="pose_detection", video_frame_callback=video_callback)
""")

# 🔁 إعادة تشغيل أي عملية Streamlit قديمة
!pkill streamlit

# ▶ تشغيل Streamlit في الخلفية
Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

# 🌍 تشغيل ngrok وإنشاء رابط خارجي
public_url = ngrok.connect(8501).public_url
print(f"\n\n🟢 افتح الرابط التالي للوصول إلى التطبيق: {public_url}")
