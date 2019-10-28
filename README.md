# tanukiNet v1 Adaptive Learning rate version

### Abstact
- Sequential tanukiNet에 미리 설정된 threshold값에 따라 Learning rate를 조작.
- 현재 알고리즘. Loss의 10%이하로 loss의 개선량이 낮아지면 learning rate를 절반으로 낮춘다.
    - 문제점. -> Learning rate가 낮아지면 속도 또한 낮아지기에 악순환이 지속.
    - 해겳책 제안
        * Relax 인자 추가 -> 업데이트 직후에는 Relax time만큼 update를 하지 않음, 기본 옵션 3.
        * threshold를 3%로 세팅. 