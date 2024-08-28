import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pandas as pd

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands  # 손 인식 모듈 초기화
hands = mp_hands.Hands()  # 손 인식 객체 생성
mp_drawing = mp.solutions.drawing_utils  # 손 랜드마크 그리기 도구 초기화

# 필터 적용 함수 정의
def apply_filter(frame, filter_type):
    """필터 타입에 따라 프레임에 필터를 적용하는 함수"""
    if filter_type == 1:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 필터 1: 그레이스케일 필터 적용
    
    elif filter_type == 2:
        # 필터 2: 카툰 효과 필터 적용

        # Step 1: 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: 적응형 임계값을 사용한 가장자리 검출
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=9, C=9)
        
        # Step 3: 원본 이미지에 양방향 필터를 적용하여 색상을 부드럽게 처리
        # sigmaColor와 sigmaSpace 값을 조정하여 경계를 더 선명하게 유지
        color = cv2.bilateralFilter(frame, d=9, sigmaColor=1, sigmaSpace=2)
        
        # Step 4: 가장자리 검출 결과와 색상을 결합하여 카툰 효과를 생성
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        return cartoon
    return frame  # 필터 타입이 1~2가 아니면 원본 프레임 반환

# 손가락 각도 계산 함수
def calculate_angle(a, b, c):
    """세 점(a, b, c) 사이의 각도를 계산하여 반환"""
    a = np.array(a)  # 점 a의 좌표를 numpy 배열로 변환
    b = np.array(b)  # 점 b의 좌표를 numpy 배열로 변환 (중심점)
    c = np.array(c)  # 점 c의 좌표를 numpy 배열로 변환
    ba = a - b  # 점 b에서 점 a로의 벡터
    bc = c - b  # 점 b에서 점 c로의 벡터
    cosangle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  # 코사인 각도 계산
    degree = np.degrees(np.arccos(cosangle))  # 코사인 값을 각도로 변환
    return degree  # 각도를 반환

# 손동작 데이터를 CSV 파일에 저장하는 함수
def data_to_csv(data, filename='hand_data.csv'):
    """손동작 데이터를 CSV 파일에 저장하는 함수"""
    file_exists = os.path.isfile(filename)  # 파일이 이미 존재하는지 확인
    with open(filename, 'a', newline='') as file:  # 파일을 추가 모드로 열기
        writer = csv.writer(file)
        if not file_exists:
            # 처음 실행 시 헤더를 작성
            writer.writerow(['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'label'])
        writer.writerow(data)  # 데이터를 파일에 작성

# KNN 모델 학습 코드 (CSV로부터 데이터를 읽어와 모델을 학습)
df = pd.read_csv('hand_data.csv')  # CSV 파일로부터 데이터를 읽어옴
X_data = df.iloc[:, :-1]  # 입력 데이터 (특징 값들)
y_data = df.iloc[:, -1]  # 출력 데이터 (레이블)
X_data = X_data.to_numpy().astype(np.float32)  # numpy 배열로 변환 및 float32 타입으로 캐스팅
y_data = y_data.to_numpy().astype(np.float32)  # numpy 배열로 변환 및 float32 타입으로 캐스팅
knn_model = cv2.ml.KNearest_create()  # KNN 모델 생성
knn_model.train(X_data, cv2.ml.ROW_SAMPLE, y_data)  # 모델 학습

# 웹캠을 통해 실시간으로 영상 캡처
cap = cv2.VideoCapture(0)  # 기본 웹캠(0번 카메라) 캡처 객체 생성

while cap.isOpened():  # 웹캠이 열려 있는 동안 실행
    ret, frame = cap.read()  # 웹캠에서 프레임을 읽어옴
    if not ret:  # 프레임을 읽어오지 못한 경우 루프 종료
        print("Error: Could not read frame from video source.")
        break
    
    frame = cv2.flip(frame, 1)  # 프레임을 좌우 반전 (거울 모드)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환 (MediaPipe는 RGB를 사용)
    
    results = hands.process(img_rgb)  # 손 인식 결과를 처리
    ih, iw, ic = frame.shape  # 프레임의 높이, 너비, 채널 수를 가져옴

    if results.multi_hand_landmarks:  # 손 랜드마크가 감지되었을 경우
        for hand_landmarks in results.multi_hand_landmarks:  # 각 손에 대해 처리
            finger_point = [
                [0, 1, 2], [1, 2, 3], [2, 3, 4],
                [0, 5, 6], [5, 6, 7], [6, 7, 8],
                [0, 9, 10], [9, 10, 11], [10, 11, 12],
                [0, 13, 14], [13, 14, 15], [14, 15, 16],
                [0, 17, 18], [17, 18, 19], [18, 19, 20]
            ]  # 손가락 각도를 계산하기 위한 랜드마크 포인트 인덱스
            
            finger_degrees = []  # 계산된 각도 리스트
            
            try:
                for fp in finger_point:  # 각 손가락에 대해 각도를 계산
                    if len(hand_landmarks.landmark) > max(fp):  # 유효한 랜드마크 인덱스인지 확인
                        f1 = hand_landmarks.landmark[fp[0]]  # 랜드마크 포인트 1
                        f2 = hand_landmarks.landmark[fp[1]]  # 랜드마크 포인트 2 (중심점)
                        f3 = hand_landmarks.landmark[fp[2]]  # 랜드마크 포인트 3
                        
                        f1_list = [f1.x * iw, f1.y * ih, f1.z]  # 포인트 1의 좌표
                        f2_list = [f2.x * iw, f2.y * ih, f2.z]  # 포인트 2의 좌표 (중심점)
                        f3_list = [f3.x * iw, f3.y * ih, f3.z]  # 포인트 3의 좌표
                        
                        f_degree = calculate_angle(f1_list, f2_list, f3_list)  # 각도 계산
                        finger_degrees.append(str(int(f_degree)))  # 계산된 각도를 리스트에 추가
                    else:
                        print("Error: Invalid landmark index")
                        continue

                if finger_degrees:  # 계산된 각도가 있을 경우
                    data = np.array([finger_degrees], dtype=np.float32)  # 각도를 numpy 배열로 변환
                    _, results, _, _ = knn_model.findNearest(data, 3)  # KNN 모델을 사용해 필터 예측
                    pred = int(results[0][0])  # 예측 결과 가져오기
                    
                    # 예측된 필터 타입에 따라 필터 적용
                    frame = apply_filter(frame, pred)

                    # 손 랜드마크를 프레임에 그리기
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            except IndexError as e:  # 인덱스 에러 처리
                print(f"Index error: {e}")
                continue
            except Exception as e:  # 기타 예외 처리
                print(f"Unexpected error: {e}")
                continue

    # 프레임을 화면에 표시
    cv2.imshow('Hand Gesture Filter Selection', frame)
    
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('1'):  # '1' 키 입력 시 (주먹 제스처)
        if finger_degrees:  # 각도 데이터가 있을 경우
            finger_degrees.append('1')  # 레이블 '1' 추가
            data_to_csv(finger_degrees)  # CSV 파일에 저장
        
    if key & 0xFF == ord('2'):  # '2' 키 입력 시 (보자기 제스처)
        if finger_degrees:  # 각도 데이터가 있을 경우
            finger_degrees.append('2')  # 레이블 '2' 추가
            data_to_csv(finger_degrees)  # CSV 파일에 저장
    
    if key & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

# 캡처 종료 및 모든 윈도우 닫기
cap.release()
cv2.destroyAllWindows()
