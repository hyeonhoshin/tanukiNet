# tanukiNet v1 _ distributed ver

### Abstact
- 해당 Branch는 Functional API로는 왜 Sequential한 구현이 안되는지를 실험한 Branch임
- 19-10-28 오전 3시 12분경을 기준으로, Sequential한 tanukiNet과 동일한 성능을 만드는데 성공

### 원인
- __he_normal kernel initializer가 오히려 나쁜 성능을 냄__
- 중복 선언 자체에는 문제가 없었음.
- Compile을 함수안에서 함. (3분 딥러닝 책도 이렇게 구현)
- 인수의 순서도 text형식으로 힌트가 있는 인수는 상관이 없었음.
- 가독성을 위해 dropout과 conv2d는 묶어 기록함.
- 구조, parameter의 수, activation function 모두 동일했었음.
- --optimizer 이름을 'Adam'에서 'adam'으로 변경, 함수형태로 부를때만 adam으로 쓰는듯..?-- 이건 원인 아님을 확인
- 3-> (3,3) 이 자체만으로는 원인이 아님.