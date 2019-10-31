# tanukiNet v2 - CBAM ver 2.

### Abstact
- tanukiNetv2 ver 2에 CBAM Layer 추가 버전

### 실험 방법
- CBAM Network에서 Dice loss가 잘 되지 않아서 MSE로 설정하고 epoch을 20까지
- Adaptive LR, 0.8 and relax = 3 적용
- 최상위 Batchnorm 적용

### 분석 결과
- 모름