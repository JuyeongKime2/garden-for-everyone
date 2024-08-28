import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe Hands 모듈 초기화 (인식 성능 향상을 위한 파라미터 조정)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 이미지 리스트 (넘길 이미지들)
image_files = ["image1.png", "image2.png", "image3.png"]
current_image_idx = 0  # 현재 보여줄 이미지 인덱스

# 이미지 창 크기 설정
resize_width = 480  # 원하는 가로 크기
resize_height = 640  # 원하는 세로 크기

# 웹캠에서 영상을 읽어오기 위한 설정
cap = cv2.VideoCapture(0)

# 손 이동 감지 임계값
MOVE_THRESHOLD = 50

# 손의 이전 위치 초기화
prev_hand_x = None

# 이미지 전환 쿨다운 타이머
cooldown = 2  # 이미지 전환 쿨다운 타임 (2초)
last_switch_time = time.time()

# 웹캠에서 실시간으로 프레임을 읽어오는 루프
while cap.isOpened():
    ret, frame = cap.read()  # 프레임을 읽음
    
    frame = cv2.flip(frame, 1)  # 프레임을 좌우 반전
    
    # 화면에 가이드라인 표시
    h, w, _ = frame.shape
    cv2.putText(frame, "Swipe your hand left or right!", 
            (70, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 105, 230),  # 분홍색 BGR 값
            2, 
            cv2.LINE_AA)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 이미지를 RGB로 변환
    
    results = hands.process(img_rgb)  # Mediapipe Hands를 사용하여 손 감지 및 랜드마크 추출
    
    # 손 랜드마크가 감지되었을 때
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손의 중심 위치 계산 (손목의 x좌표 사용)
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w

            # 손의 이전 위치와 현재 위치의 차이를 계산하여 움직임 감지
            if prev_hand_x is not None:
                move_distance = hand_x - prev_hand_x

                # 손이 오른쪽으로 MOVE_THRESHOLD 이상 움직이면 다음 이미지로
                if move_distance > MOVE_THRESHOLD:
                    current_time = time.time()
                    if (current_time - last_switch_time) > cooldown:
                        current_image_idx = (current_image_idx + 1) % len(image_files)
                        last_switch_time = current_time
                        print("Next Image")

                # 손이 왼쪽으로 MOVE_THRESHOLD 이상 움직이면 이전 이미지로
                elif move_distance < -MOVE_THRESHOLD:
                    current_time = time.time()
                    if (current_time - last_switch_time) > cooldown:
                        current_image_idx = (current_image_idx - 1) % len(image_files)
                        last_switch_time = current_time
                        print("Previous Image")

            # 손의 현재 위치를 이전 위치로 저장
            prev_hand_x = hand_x

            # 손의 랜드마크와 연결선을 그려줌
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        # 이미지 표시
        img = cv2.imread(image_files[current_image_idx])
        
        if img is not None:
            img_resized = cv2.resize(img, (resize_width, resize_height))
            cv2.imshow('image', img_resized)
        else:
            print(f"Error: Unable to load image {image_files[current_image_idx]}")

    # 결과 프레임을 화면에 표시
    cv2.imshow('hand', frame)
    
    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
