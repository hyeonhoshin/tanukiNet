# tanukiNet v2 candidate - nobatchnorm

### Abstact
- tanukiNetv2-codinfox ver 2에 batchnorm을 모두 제거하고 학습

### 실험 방법
- Batchnorm이 오히려 방해를 하나 싶어서 Batch norm 및 Dropout 모두 제거
- 이전 실험과의 동일성을 위해 initializer는 he_init으로

### 분석 결과
- 