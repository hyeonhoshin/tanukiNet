# tanukiNet v2 - condinfox-dice

### Abstact
- tanukiNetv2-codinfox branch에 loss를 dice로 변경하여 최종 테스트
- 학습이 느린 경향이 있었으므로, epoch은 넉넉히 30정도 줌.
- Cosine Adaptive LR을 모방하기 위해 rest간격을 짧게 하는 대신 감소 비율을 0.8로 변경
- 최상위 layer에 batch norm이 있는 이유를 모르겠어서 제거
- 빠른 학습을 위해 kernel 수 v1과 동일하도록 감소 시킴
- he_init도 loss가 정체되는 현상이 여전히 나타나서 안함.
- ** 이것 전부 효과가 없을 경우 filter 수 좀 늘린거에 adaptive lr이랑 attention이랑 MSE만 추가해서 제출할거임. 으으 batch norm쓰면 오히려 성능이 떨어지는거 같기도... **

### 분석 결과
- dice를 적용하니, 오히려 val loss등이 감소하거나 하지 않음.
- 폐기.