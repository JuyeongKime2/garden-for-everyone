# Hand Gesture Filter Selection using OpenCV and MediaPipe

## 프로젝트 개요
이 프로젝트는 웹캠을 사용하여 손동작을 인식하고, 손가락 각도를 기반으로 필터를 적용하는 애플리케이션입니다. OpenCV와 MediaPipe를 사용하여 손 랜드마크를 인식하고, 사용자의 제스처에 따라 실시간으로 필터를 선택하여 적용합니다. 또한, 손동작 데이터를 CSV 파일에 저장하고, KNN 알고리즘을 사용하여 제스처를 분류합니다.

## 기능
- **손동작 인식**: MediaPipe를 사용하여 실시간으로 손 랜드마크를 인식합니다.
- **필터 적용**: 사용자의 제스처에 따라 필터(그레이스케일, 카툰 효과)를 적용합니다.
- **손동작 데이터 저장**: 손가락 각도 데이터를 CSV 파일에 저장합니다.
- **KNN 모델 학습**: 저장된 손동작 데이터를 사용하여 KNN 모델을 학습시킵니다.
- **제스처 기반 필터 선택**: 학습된 KNN 모델을 사용하여 제스처에 따라 필터를 자동으로 적용합니다.

## 요구사항
이 프로젝트를 실행하기 위해서는 다음과 같은 라이브러리가 필요합니다:

- Python 3.7 이상
- OpenCV
- MediaPipe
- NumPy
- Pandas

## 설치 방법

1. Python 가상환경 생성 및 활성화:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    venv\Scripts\activate     # Windows
    ```

2. 필요한 라이브러리 설치:
    ```bash
    pip install opencv-python mediapipe numpy pandas
    ```

3. 프로젝트 클론:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

4. 손동작 데이터를 위한 CSV 파일 준비:
    프로젝트 디렉토리에 `hand_data.csv` 파일이 있어야 합니다. 처음 실행 시 자동으로 생성되며, 기존 데이터가 있는 경우 해당 파일을 사용할 수 있습니다.

## 사용법

1. 웹캠을 통해 실시간 영상 캡처를 시작합니다:
    ```bash
    python main.py
    ```

2. 프로그램이 실행되면, 사용자는 다음과 같은 손 제스처를 통해 필터를 적용할 수 있습니다:
    - **'1' 키**: 주먹 제스처로 필터를 학습시킬 수 있으며, CSV 파일에 저장됩니다.
    - **'2' 키**: 보자기 제스처로 필터를 학습시킬 수 있으며, CSV 파일에 저장됩니다.
    - **실시간 제스처 인식**: 손동작이 인식되면 학습된 KNN 모델에 따라 필터가 자동으로 적용됩니다.

3. 프로그램 종료:
    - **'q' 키**를 누르면 프로그램이 종료됩니다.

## 예제

![Hand Gesture Example](example.gif)

위 GIF는 주먹 제스처와 보자기 제스처를 사용하여 필터를 적용하는 예시입니다.

## 주의사항
- 손동작 인식 정확도를 높이기 위해 충분한 학습 데이터를 수집하여 CSV 파일에 저장하는 것이 중요합니다.
- 다양한 조명 조건에서 손이 잘 인식될 수 있도록 환경을 조정하세요.

## 기여
이 프로젝트에 기여하고 싶다면, 이 리포지토리를 포크하고 풀 리퀘스트를 보내주세요. 모든 종류의 기여를 환영합니다!

## 라이선스
이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
