# tanukiNet v2 - CBAM

### Abstact
- tanukiNetv2 ver 2에 CBAM Layer 추가 버전

### 실험 방법
- 일단 Conv에만 CBAM을 추가
- tanukiNetv2 ver 2에서 epoch 9에서 어떤 결과도 나오지 않은 것을 바탕으로, epoch을 30까지 늘려봄.

### 분석 결과
- 중간에 중복된 선에 의해 열화가 심하게 일어남
- 그림자에서의 성능은 거의 비슷
- val_loss는 epochs = 25이후로 거의 변하지 않음.
- Competion metric은 점점 증가.