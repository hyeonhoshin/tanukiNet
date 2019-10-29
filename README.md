# tanukiNet v2

### Abstact
- 필터 수 증가 from filter up
- merge layers는 제거
- loss function은 iou추가, [출처](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044)
- Adaptive LR추가
- loss는 음... MSE일단.(그 다음에는 )
- history reader 추가

### 분석 결과
#### Try 1.
- epoch = 20, no merge, loss = MSE
- 

#### Try 2.
- BAM and CBAM Layer 후보군 추가 ( branch 생성 필요 )