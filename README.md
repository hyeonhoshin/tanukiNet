# tanukiNet v1 _ deepRobust

### Abstact
- 해당 Branch는 deeper branch에서 파생되었으며, 해당 Branch에서 overfitting이 관측되어 이를 해결하기 위해 filter의 수를 줄인 버전이다.

### 분석
- 영상 : deeper보다는 그림자에서의 성능은 좋아졌으나, 도로 위 낙서에서는 오히려 성능이 떨어졌다. epochs = 10
- epochs수를 loss를 보고 최적화 시키면 좋을듯하다.
