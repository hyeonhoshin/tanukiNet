# Modified Unet

### Abstract
- unet과 비슷하게 Merge layer를 하나 추가.
- 그외에는 tanukiNet v1과 동일.
- 기존 실험 시, 학습때 regulization을 안해줘서 결과가 왜곡되었을 가능성이 큼. -> 실험 중

### 실험결과 
- Kernel수 약 2배
    * loss의 감소 속도는 아주 미세하게 느려졌으나, val_loss가 epoch 2의 0.0243부근에서 더이상 떨어지지 않음
    * 학습 속도가 tanukiNet v1과 비슷해짐.
    * 영상 : 실제로 아무것도 나오지 않음
- 이번에는 he_init을 제거해보려고 함.
    * 확실히 epoch 1에서의 loss감소 속도가 빨라짐
    * 그러나 최종적인 val_loss는 여전히 감소하지 않음.