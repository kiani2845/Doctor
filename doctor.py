import cv2
import numpy as np
import librosa
import wave
#import gradio as gr
import pyaudio
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
import flet as ft
import openai
import threading
from PIL import Image
# Import Dependencies
import yaml
from joblib import dump, load
from joblib import load
import pandas as pd
#import numpy as np
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
import os
import torch
from collections import Counter
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
global symptoms

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# مسیر ویدیو
  
video_path = 'recorded_video.avi'
if video_path == '' and video_path != 'recorded_video.avi':
    pass
else:    
# باز کردن ویدیو
    cap = cv2.VideoCapture(video_path)

# چک کردن اینکه ویدیو باز شده است یا نه
if not cap.isOpened():
    print("خطا در باز کردن ویدیو.")
else:
    # تعداد فریم‌ها
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # فریم‌ها از 1 تا 10
    for i in range(1, 11):
        # محاسبه فریم
        frame_number = i - 1  # برای اینکه شمارش از 0 شروع می‌شود
        
        if frame_number < frame_count:
            # خواندن فریم
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # ذخیره فریم به صورت JPG
                cv2.imwrite(f'000{i}.jpg', frame)
                print(f'فریم {i} ذخیره شد.')
            else:
                print(f'خطا در خواندن فریم {i}.')
        else:
            print(f'فریم {i} موجود نیست.')

# آزاد کردن ویدیو
cap.release()
    
video_dir = "./"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))   

inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
frame = show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
global symptom_list
# Function to make Inference
def predict_disease_from_symptom(symptom_list):
    symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing','shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue','muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue','weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss','cold','flu','restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough','high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration','indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite','pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever','yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach','swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation','redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs','fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool','irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs','swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties','excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech','knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness','spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell','bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching','toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium','red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes','increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration','visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma','stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1','blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples','blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails','blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    
    # Set value to 1 for corresponding symptoms
    for s in symptoms:
        symptoms[s] 
    
    # Put all data in a test dataset
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))
    
    # Load pre-trained model
    clf = load(str("./saved_model/random_forest.joblib"))
    global result
    result = clf.predict(df_test)
    
    # Cleanup
    del df_test
    
    return f"{result[0]}"
def myWebcam():
    
# Initialize webcam, cap is the object provided by VideoCapture
    cap = cv2.VideoCapture(0)

    while True:
    # It contains a boolean indicating if it was sucessful (ret)
    # It also contains the images collected from the webcam (frame)
        ret, frame = cap.read()
    
        cv2.imshow('Our Webcam Video', frame)
    
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break
        
# Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()   

# تابع برای تولید تصویر طراحی
def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

# تابع برای پردازش تصویر و پیش‌بینی
def predict_from_video(frame):
    # استخراج ویژگی‌های تصویر
    img_array = extract_image_features(frame)
    prediction = image_model.predict(img_array)
    return "Flu" if prediction > 0.5 else "Cold"

# تابع برای بروزرسانی تصویر
def update_sketch(page, image_container):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        sketched_image = sketch(frame)
        _, buffer = cv2.imencode('.png', sketched_image)
        img_bytes = buffer.tobytes()
        
        # بروزرسانی تصویر در رابط کاربری
        #image_container.image = ft.Image(src=ft.ImageSource.from_bytes(img_bytes))
        page.update()
    cap.release()

# تنظیم کلید API OpenAI
openai.api_key = '81793d4f-300c-4ca9-b143-97f263c5d3e0'

# تابعی برای استخراج ویژگی‌های صوتی
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# تابعی برای پردازش تصویر
def extract_image_features(image):
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    image = image.astype("float") / 255.0
    return np.expand_dims(image, axis=0)

# ساخت مدل LSTM برای صدا
def create_audio_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ساخت مدل CNN برای تصویر
def create_image_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# تابع تولید متن با استفاده از OpenAI API
def generate_text(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# تابع ضبط ویدئو
def record_video(filename, duration=50):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    print("Recording video...")
    start_time = cv2.getTickCount()
    while int((cv2.getTickCount() - start_time) / cv2.getTickFrequency()) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    print("Finished recording.")
    cap.release()
    out.release()

# تابع ضبط صدا
def record_audio(filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5  # مدت زمان ضبط صدا
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("Recording audio...")
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording audio.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# تابع نمایش ویدیو زنده
def display_video(page, video_container):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # پیش‌بینی از تصویر
        prediction = predict_from_video(frame)
        cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.png', frame)
        img_bytes = buffer.tobytes()

        # بروزرسانی تصویر در رابط کاربری
        #video_container.image = ft.Image(src=frame)
        page.update()
    cap.release()


# Webcam for symptom prediction
def predict_from_webcam(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_array = preprocess_image(frame)
        prediction = model.predict()
        
        cv2.putText(frame, f"Prediction: {symptoms}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# ساخت رابط کاربری با Flet
def main(page: ft.Page):
    page.title = "Health Predictor Doctor(by hamed kiani)"
    page.bgcolor = ft.colors.YELLOW  # تنظیم رنگ پس‌زمینه
    # عنصر نمایش تصویر
    image_container = ft.Container()
    video_container = ft.Container()

    # شروع بروزرسانی تصویر و نمایش ویدیو در تردهای جداگانه
    threading.Thread(target=update_sketch, args=(page, image_container), daemon=True).start()
    threading.Thread(target=display_video, args=(page, video_container), daemon=True).start()

    # ورودی‌ها
    audio_file_input = ft.FilePicker()
    image_file_input = ft.FilePicker()
    prompt_input = ft.TextField(label="Enter your prompt for text generation(You dont have any disease)", multiline=True, expand=True)
    output_text = ft.TextField(label="Generated Response", read_only=True, multiline=True, expand=True)
    # Put all data in a test dataset
        #global df_test
        #global clf
        #global result
        #global webcam_butoon
        #df_test = pd.DataFrame(columns=list(symptoms.keys()))
        #clf = load(str("./saved_model/random_forest.joblib"))
        #result = clf.predict(df_test)
    #global df_test
    #global clf
    #global result
    global webcam_butoon
    #global symptoms

    symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing','shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue','muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue','weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss','cold','flu','restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough','high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration','indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite','pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever','yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach','swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation','redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs','fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool','irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs','swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties','excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech','knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness','spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell','bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching','toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium','red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes','increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration','visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma','stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1','blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples','blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails','blister', 'red_sore_around_nose', 'yellow_crust_ooze']
        
    for s in symptoms:
        Counter(symptoms) == 1
    
    
    webcam_button = ft.ElevatedButton("Start Webcam Prediction", on_click=lambda e: predict_from_webcam(load(str("./saved_model/random_forest.joblib")).predict(pd.DataFrame(columns=list(s[0])))))

    # دکمه‌ها
    record_video_button = ft.ElevatedButton("Record Video", on_click=lambda e: record_video('recorded_video.avi', duration=10))
    record_audio_button = ft.ElevatedButton("Record Audio", on_click=lambda e: record_audio('audio.wav'))
    predict_button = ft.ElevatedButton("Predict from Audio", on_click=lambda e: predict_audio())
    predict_from_webCam = ft.ElevatedButton("Predict from Webcam", on_click=lambda e: predict_image())
    predict_image_button = ft.ElevatedButton("Predict from Image", on_click=lambda e: predict_image())
    generate_button = ft.ElevatedButton("Generate Text", on_click=lambda e: generate_text_response())
# دکمه "Open My Webcam"
    open_webcam_button = ft.ElevatedButton("Open My Webcam", on_click=lambda e: myWebcam())

# افزودن عناصر به صفحه
    page.add(video_container, record_video_button, record_audio_button, open_webcam_button, predict_button, image_file_input, predict_image_button, prompt_input, generate_button, output_text, image_container,webcam_button)

    # توابع پیش‌بینی
    def predict_audio():
        #global symptoms
        features = extract_audio_features('audio.wav')
        features = features.reshape(1, -1, 1)
        prediction = audio_model.predict(features)
        result = symptoms
        page.add(ft.Text(f"Audio Prediction: {result}"))
    def record_video(filename, duration=100):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

        print("Recording video...")
        start_time = cv2.getTickCount()
        while int((cv2.getTickCount() - start_time) / cv2.getTickFrequency()) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        print("Finished recording.")
        cap.release()
        out.release()
   

    def predict_image():
       #global symptoms
        if frame:
            image_file = frame  # استفاده از مسیر فایل
            img_array = extract_image_features(cv2.imread('0009.jpg'))
            prediction = image_model.predict(img_array)
            result = prediction
            page.add(ft.Text(f"Image Prediction: {result}"))

    def generate_text_response():
        prompt = prompt_input.value
        generated_response = generate_text(prompt)
        output_text.value = generated_response
        page.update()

    # افزودن عناصر به صفحه
    page.add(video_container, record_video_button, record_audio_button, predict_button, image_file_input, predict_image_button, prompt_input, generate_button, output_text, image_container,webcam_button)

audio_model = create_audio_model((13, 1))  
image_model = create_image_model((64, 64, 3))  

ft.app(target=main)
