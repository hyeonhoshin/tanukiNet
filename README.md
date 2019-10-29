# tanukiNet v2

### Abstact
- 필터 수 증가 from filter up
- merge layers는 제거
- loss function은 iou추가, [출처](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044)
- Adaptive LR추가
- loss는 음... MSE일단.
- history reader 추가

### Update
#### ver 2.
- Optimizing. epoch = 9에서 멈춤

### 분석 결과
#### ver 1.
- epoch = 20, no merge, loss = MSE
- epoch = 9쯤에서 val_iou_loss_core가 가장 높다.

#### ver 2.
- 영상 상에 어떤 것도 보이지 않음.