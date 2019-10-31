# CBAM-dice20

### Abstact
- tanukiNetv2-CBAM이 가능성을 보여주었으므로 그나마 가능성을 보여준 dice loss의 경우와 통합하여 그 결과를 평가
- decay = 0.8, rest는 3마다 일어남.
- epoch을 dice branch의 두 배인 20으로 끌어올림

### 분석 결과
- epoch을 올리니까 결과가 더 잘나온다. -> 30으로 끌어올려 보자