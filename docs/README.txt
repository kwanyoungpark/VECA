COGNIADemoEnv (Good/bad object setup)

Task 설명

방 안에 아기와 각기 다른 크기 3개(S,M,L), 자성 2개(N,S) 총 6개의 디스크가 있다.
디스크들을 크기 순서대로 쌓아서 6개의 디스크를 모두 쌓는 것이 목표이다.
자성이 같은 디스크는 올려놓을 수는 있지만 척력으로 인해 물리적으로 튕겨져 나오게 된다.
크기가 더 큰 디스크는 작은 디스크 위해 올려놓을 수는 있지만 대부분 무게중심 때문에 무너진다.

Reward
제대로 쌓여진 디스크 개수 - 1  (범위 : 0 ~ 5)

Observation 
Vision : 범위 [0, 1], size = (6, 84, 84)
reward: Reward 설명란 참조

Auxillary information
pos: 아기 좌표계 기준 디스크들의 위치
grab: 어떠한 물체를 지금 잡고 있는지

Action
Continuous 4-dim + Discrete 1-dim (총 5-dim)
action[0], action[1] 은 아기의 이동방향 (앞/뒤, 왼/오른쪽)
action[2], action[3] 은 아기 목의 회전 방향 (위/아래, 왼/오른쪽. 시야가 제한되어 있기 때문에 필요하다)
action[4] 는 Discrete action으로 0 이하의 값이 들어오면 아무것도 하지 않는다.
0 이상의 값이 들어오면 현재 물체를 잡고 있지 않다면 grab으로, 잡고있는 물체가 있다면 release 액션으로 판정된다.

Grab 액션을 취했을 때 어느 물체를 잡을지의 기준 :
1) 아기의 시야각 30도 안에 있어야 함
2) 다른 물체에 완전히 가려지면 (그 물체를 아기가 관측할 수 없으면) 안됨.
3) 아기와의 거리가 3유닛 이하여야 함.
4) 만약 그러한 물체가 여러개 있을 경우 아기의 시야 중앙에 제일 가까운 물체를 선택
5) 만약 그러한 물체가 없다면 No-action으로 간주된다.

Release: 아기의 시점 기준으로 3 유닛 앞에 물체를 놓는다. 만약 그 놓는 위치가 환경 밖이라면 환경 안으로 들어오도록 좌표가 수정된다.

Arguments (DiskTowerEnv.py에서 변경)

"-debug" / "-train" : debug 모드일 시 AI가 조작 불가(플레이어가 조작 가능) train 모드일 때 AI가 조작 가능 (default: train)
"-record" : 녹화용 Top-down view 화면을 Observation을 보낼 때 같이 "Recimg" 키로 보냄.

*이 TUTORIAL은 Windows 기준입니다. VECA는 (적어도 LocalMPWrapper와 환경은) Windows 환경에서 실행하는 것을 추천드립니다. *

파일을 다운로드하면, LocalMPWrapper, COGNIADemoApp, PythonWrapper 3개의 폴더가 있습니다.

1. Setup

1) LocalMPWrapper에 있는 DiskTowerEnv.py에서 7번째 줄을 고쳐주세요(경로 변경):

~ exec_str = "(다운로드 받은 경로)\\BabyMindDiskTowerApp\\BabyMindDG.exe"
(만약 LocalMPWrapper랑 환경 경로를 유지하신다면 안고치셔도 됩니다)

2) LocalMPWrapper에 있는 COGNIADemoEnv.py에서 10번째 줄을 고쳐주세요(알고리즘 쪽 ip 변경):

~ ip = (GPU 서버 IP)

만약 알고리즘 또한 같은 컴퓨터에 있다면 localhost(127.0.0.1)로 설정해주세요.

3) PythonWrapper 안의 test.py에서 4번째줄을 고쳐주세요(포트설정, 기본은 8872로 설정되어 있습니다):

~ port = (GPU 서버 포트)

2. 실행: Single environment

명령프롬포트(cmd)가 2개 필요합니다. (Python wrapper를 가지고 있는 서버 측 cmd, LocalMPWrapper를 가지고 있는 로컬 측 cmd)

0) test.py에서 4번째 줄을 고쳐주세요(environment 개수)

~ NUM_ENVS = 1

1) 첫번째 프롬포트에서 PythonWrapper에 있는 test.py를 실행해주세요:

>> python test.py

그리고 "0.0.0.0"이 나타날때 까지 기다려주세요 (연결 waiting time)

2) 그 다음 LocalMPWrapper에 있는 DiskTowerEnv.py를 실행시켜주세요:

>> python COGNIADemoEnv.py (GPU 서버 포트) 7777 1
P.S. 7777은 (GPU 서버 포트)와 다른 숫자이면서 valid한 Local 포트 번호이면 됩니다.

그러면 연결이 되면서 random action을 할 것입니다.

3. 실행: 8 cocurrent environment (A3C등을 위한 parallel execution)
마찬가지로 명령프롬포트(cmd)가 2개 필요합니다.

0) test.py에서 4번째 줄을 고쳐주세요(environment 개수)

~ NUM_ENVS = 8

1) 첫번째 프롬포트에서 PythonWrapper에 있는 test.py를 실행해주세요:

>> python test.py

그리고 "0.0.0.0"이 나타날때 까지 기다려주세요 (연결 waiting time)

2) 그 다음 LocalMPWrapper에 있는 DiskTowerEnv.py를 실행시켜주세요:

>> python DiskTowerEnv.py (GPU 서버 포트) 7777 8
P.S. 7777은 [N, N+8)이 (GPU 서버 포트)와 겹치지 않으면서 valid한 Local 포트 번호 N이면 됩니다.

그러면 8개의 environment에 parallel로 연결이 되면서 random action을 할 것입니다.

4. 녹화 (with hand-crafted algorithm)

0)  LocalMPWrapper에 있는 DiskTowerEnv.py에서 7번째 줄을 각주처리하고 8번째 줄을 각주해제해주세요. (record argument 추가)

~ #args = ["-train", "-timeout", "-1", "-notactile"]
~ args = ["-train", "-timeout", "-1", "-notactile", "-record"]

1) 첫번째 프롬포트에서 PythonWrapper에 있는 handcrafted_policy.py를 실행해주세요:

>> python handcrafted_policy.py

그리고 "0.0.0.0"이 나타날때 까지 기다려주세요 (연결 waiting time)

2) 그 다음 LocalMPWrapper에 있는 DiskTowerEnv.py를 실행시켜주세요:

>> python DiskTowerEnv.py (GPU 서버 포트) 7777 1
P.S. 7777은 (GPU 서버 포트)와 다른 숫자이면서 valid한 Local 포트 번호이면 됩니다.

그러면 연결이 되면서 hand-crafted policy를 통해 action을 할 것입니다.
끝나고 나면 녹화본이 record.mp4로 저장됩니다.


