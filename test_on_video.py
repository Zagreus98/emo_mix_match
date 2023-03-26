import cv2
import numpy as np
from models import emotion_model
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import mediapipe as mp

detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

emo_dict = {0: 'Surprise',
            1: 'Fear',
            2: 'Disgust',
            3: 'Happy',
            4: 'Sad',
            5: 'Angry',
            6: 'Neutral'}

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([
    T.Lambda(lambda x: cv2.resize(x, (100, 100))),
    T.ToTensor(),
    normalize
])

model_path = './result/exp012_mixup/model_best.pth.tar'
score_thr = 0.2
model = emotion_model.Model(model_path='./models/resnet18_msceleb.pth')
model.eval()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])


# Run model with webcam or on a custom video
cap = cv2.VideoCapture('video00.mp4')
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # CROP THE FACE
    h, w = image.shape[:2]
    predictions = detector.process(image[:, :, ::-1])
    if predictions.multi_face_landmarks:
        for prediction in predictions.multi_face_landmarks:
            pts = np.array([(pt.x * w, pt.y * h)
                            for pt in prediction.landmark],
                           dtype=np.float64)
            bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
            bbox = np.round(bbox).astype(np.int32)

        x = bbox[0, 0]
        width = bbox[1, 0]
        y = bbox[0, 1]
        height = bbox[1, 1]
        bbox = np.round(bbox).astype(np.int).tolist()
        cv2.rectangle(image, tuple(bbox[0]), tuple(bbox[1]), (255, 0, 0), 2)
        crop_img = image[y:height + 1, x:width + 1, :].copy()

        # Predict emotion
        input_crop = transform(crop_img)
        scores = model(input_crop.unsqueeze(0))
        scores = F.softmax(scores, dim=1)
        scores = scores[0].data.cpu().numpy()
        pred = np.argmax(scores)
        conf = scores[pred]
        # Draw emotion
        if scores[pred] > score_thr:
            emotion_class = emo_dict[pred]
            cv2.putText(image, f"{emotion_class}:{conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, f"{emotion_class}:{conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', image[..., ::-1])
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
