# tanukiNet v2 - CBAM ver 2.

### Abstact
- tanukiNetv2 ver 2에 CBAM Layer 추가 버전

### 실험 방법
- CBAM Network에서 Dice loss가 잘 되지 않아서 MSE로 설정하고 epoch을 20까지
- Adaptive LR, 0.8 and relax = 3 적용
- 최상위 Batchnorm 적용
- Draw lane에 median 및 저장량을 6으로 올림

### 분석 결과
- 밝은 구간에서는 인식하는 구간을 매우 두껍게 잡음
- 그러나 오히려 Shallow에서는 그 성능이 하락함.
- 차선 노이즈에 대해서는 나은 성능을 보임