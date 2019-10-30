# tanukiNet v2 - condinfox ver 2

### Abstact
- tanukiNetv2-CBAM에 깊은 모델을 더 잘 학습시키기 위한 세가지 기법 동시 적용
- 적용 방법은 [codinfox 블로그](https://buomsoo-kim.github.io/keras/2018/05/05/Easy-deep-learning-with-Keras-11.md/) 참조

### 실험 방법
- Dropout을 Batch norm으로 변경
- 학습이 너무 오래걸려서 nin은 제거

### ver 2. 실험 방법 추가
- 계층이 서서히 증가/ 감소하도록 변경
- LR의 decay 비율은 0.8, 휴식 시간은 3 epoch를 주어 선형보다는 부드럽게 lr이 감소하도록
- 총 epoch는 30으로 상승 뒤 관찰

### ver 2. 분석 결과
- loss가 변하지 않는 것으로 보아 epoch 3부터 거의 학습이 안되고 있는것으로 추측됨.
- epoch = 3으로 내려서 그 결과만 확인.