# Atrous laon branch

## 개요
- Atrous Net 사용 후, 부적합함이 판단
- 그러나 Laon People 페이지와 Model.summary()를 보고 dilated의 개념을 이해하여 약간 변형코자함.
- block하나를 제거하고(너무 데이터 손실이 많으므로), 대신 커널 갯수를 원본과 동일하게 키움.
- 물론 맨 마지막 conv같은 경우는 2048개는 너무 오래걸릴 듯하여 1024개로 세팅

## 실험결과
- epoch = 2 즈음에서 overfitting 보임. 따라서 이때의 값으로 lane detect시도.
