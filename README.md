# tanukiNet v2-theta

### Abstact
- tanukiNetv1의 개선 버전
- train_model.py이나 draw_lanes.py는 --help 옵션으로 상세 사용법을 확인가능
- 방향 추측 함수 추가(get_dirs.py)

### 적용 기법
#### Adaptive LR
- Learning rate를 loss 감소량과 시간 변화에 따라서 감소시켜, Local minimum에서 진동이 일어나지 않도록 설정.
#### CBAM Layer 추가
- 이전 Conv layer에서 생성된 feature layer를 강화하는 기능.
#### Weighted Stabilizer
- 차선 생성 시, 각 프레임에 가중치를 두어 현재 인식 중인 Lane을 유추함.
#### 방향 추측
- Canny연산과 P-Hough transform을 이용하여 Line을 추출

### 사용 방법
#### 0. Prequisite
* python 3.6.9
* numpy
* moviepy
* pickle
* keras
* Pillow
* scikit-learn
* opencv-python

If you are using Anaconda, I recommend to use 'tanukiNet_env.yml' for generating a copy of my python environment.

#### 1. Git 저장소 다운로드
<pre><code>git clone https://github.com/Tanukimong/capstone.git</code></pre>

#### 2. tanukiNetv2 branch로 전환
<pre><code>git checkout tanukiNetv2</code></pre>

#### 3. draw_lanes.py 실행
- draw_lanes.py는 같은 폴더내에 존재하는 tanukiNetv2.json(신경망 구조 파일)과 tanukiNetv2.h5(신경망 내 가중치 파일)을 읽어들여 Lane detection을 시행
<pre><code>python draw_lanes.py -i input_filename -o output_filename</code></pre>

#### Roles of files

|      File      |                                  Role                                  |
|:--------------:|:----------------------------------------------------------------------:|
|  draw_lanes.py |      Draw detected lanes using tanukiNetv2.json and tanukiNetv2.h5     |
|   output.mp4   |             Lane detection result from challenge_video.mp4             |
|  tanuki_ml.py  |                     tanukiNetv2's stucture is here                     |
| train_model.py | train tanukiNetv2 by tanuki_train.p and get test loss by tanuki_test.p |

#### Data sets
These are originated from [CULane Project](https://xingangpan.github.io/projects/CULane.html)

|      File      |            Description            |     Link     |
|:--------------:|:---------------------------------:|:------------:|
| tanuki_train.p | Train set, Resolution = (273, 98) | [Google drive](https://drive.google.com/open?id=1IA7znH0iWGnarn74MxIRNHC1WrUSI9wk) |
|  tanuki_test.p |  Test set, Resolution = (273, 98) | [Google drive](https://drive.google.com/open?id=1zOBfQBksbFk2MTfANOaN7KlnDPe4xRyb) |