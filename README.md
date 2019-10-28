# Atrous laon branch

## 개요
- Atrous Net 사용 후, 부적합함이 판단
- 그러나 Laon People 페이지와 Model.summary()를 보고 dilated의 개념을 이해하여 약간 변형코자함.
- block하나를 제거하고(너무 데이터 손실이 많으므로), 대신 커널 갯수를 원본과 동일하게 키움.

## 실험결과
- epoch = 2 즈음에서 overfitting 보임. 따라서 이때의 값으로 lane detect시도.
- 영상 자체도 의미없게 나오는군. 음
