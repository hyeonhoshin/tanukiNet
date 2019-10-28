# tanukiNet v1 _ Deeper ver

### Abstact
- 해당 Branch는 추가로 6개의 Layer(Enc = 3, Dec =3)를 추가했을때 성능 추이를 보기 위한 Branch임
    * 해상도가 낮아 core layer가 벡터가 나와버려서 & epoch당 7분이 걸려서
    * pooling을 제외하고 conv와 deconv만 추가.
- 19-10-28 오전 3시 12분경을 기준으로, Sequential한 tanukiNet과 동일한 성능을 만드는데 성공
- train_model.py를 실행 시, history.p에 history파일 저장되며, tanukiNetv1.json, tanukiNetv1.h5에 파일이 저장됨.

### 분석
- Validation loss 기준 기존 tanukiNetv1 + Adaptive LR + epoch 20보다 더 빠르게 val_loss가 감소하는 경향을 보임
- epochs 7만에 기존 tanukiNet보다 val_loss가 낮아짐. 8이후부터는 val_loss 증가
- Adaptve LR 1%, relax = 5정도로 epoch 20돌리면 성능이 더 오르지 않을까 기대됨
- 영상은 좋으나 그림자 아래에서 아무것도 안보이며, 작은 변화에도 민감 -> 아마 overfitting
