# Dell Pro Max with GB10 듀얼 노드 Llama 405B 추론 완전 가이드

이 가이드는 Dell Pro Max with GB10 2대를 사용하여 Llama 3.1 405B 모델을 분산 추론하는 전체 과정을 설명합니다. PC를 처음 구입한 초보자도 따라할 수 있도록 모든 단계를 상세히 기술합니다.

## 목차

1. [사전 요구사항](#1-사전-요구사항)
2. [하드웨어 연결](#2-하드웨어-연결)
3. [네트워크 설정 및 IP 확인](#3-네트워크-설정-및-ip-확인)
4. [SSH 비밀번호 없이 연결 설정](#4-ssh-비밀번호-없이-연결-설정)
5. [NCCL 빌드 및 테스트](#5-nccl-빌드-및-테스트)
6. [vLLM 환경 구성](#6-vllm-환경-구성)
7. [Ray 클러스터 실행](#7-ray-클러스터-실행)
8. [405B 모델 추론 실행](#8-405b-모델-추론-실행)
9. [트러블슈팅 완전 가이드](#9-트러블슈팅-완전-가이드)
10. [정리 및 종료](#10-정리-및-종료)
11. [빠른 참조 카드](#빠른-참조-카드)
12. [부록 A: 성공적인 테스트 결과 예시](#부록-a-성공적인-테스트-결과-예시)
13. [부록 B: 실패한 설정들 (참고용)](#부록-b-실패한-설정들-참고용)
14. [부록 C: 환경 변수 전체 목록](#부록-c-환경-변수-전체-목록)

## 필독! 성공을 위한 핵심 체크리스트

> 시작하기 전에 반드시 확인하세요:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    ⚠️ 절대 실수하면 안 되는 3가지 ⚠️                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1️⃣ vLLM 버전은 반드시 0.11을 사용하세요!                                │
│     ✅ 사용: nvcr.io/nvidia/vllm:25.11-py3                              │
│     ❌ 사용금지: nvcr.io/nvidia/vllm:25.12.post1-py3 (KV cache 버그)    │
│                                                                         │
│  2️⃣ 네트워크 인터페이스는 소문자 enp1s0f...를 사용하세요!                │
│     ✅ 사용: enp1s0f0np0                                                │
│     ❌ 사용금지: enP2p1s0f0np0 (대문자 P)                                │
│                                                                         │
│  3️⃣ shm-size는 최소 10GB 이상으로 설정하세요!                           │
│     ✅ 사용: --shm-size 10.24g                                          │
│     ❌ 사용금지: --shm-size 1g (프로세스 간 통신 실패)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. 사전 요구사항

### 하드웨어

- Dell Pro Max with GB10 2대 (각각 NVIDIA GB10 GPU 1개 탑재)
- QSFP 케이블 1개 이상 (200Gbps 대역폭)
- 각 PC에 모니터, 키보드 연결 (초기 설정용)

### 소프트웨어 (사전 설치 필요)

- Ubuntu 24.04 LTS (기본 OS)
- Docker 및 NVIDIA Container Toolkit
- CUDA Toolkit
- OpenMPI (libopenmpi-dev)

### 계정 정보

> 이 가이드에서는 사용자 계정을 `tester`로 가정합니다. 본인의 계정명으로 대체하세요.

## 2. 하드웨어 연결

### 2.1 QSFP 케이블 연결

Dell Pro Max with GB10 후면에는 ConnectX-7 네트워크 카드의 QSFP 포트가 있습니다.

```text
PC1 (Head Node)                    PC2 (Worker Node)
+------------------+               +------------------+
|                  |               |                  |
|   [QSFP Port 1]--|---케이블----|--[QSFP Port 1]   |
|   [QSFP Port 2]  |               |  [QSFP Port 2]  |
|                  |               |                  |
+------------------+               +------------------+
```

**중요:**

- 케이블 1개로도 전체 대역폭(200Gbps)을 달성할 수 있습니다

### 2.2 전원 연결 및 부팅

1. 각 PC에 전원 케이블 연결
2. 전원 버튼을 눌러 부팅
3. 로그인 화면에서 계정으로 로그인

## 3. 네트워크 설정 및 IP 확인

### 3.1 네트워크 인터페이스 상태 확인

PC1에서 터미널 열기 (Ctrl+Alt+T)

```bash
# RDMA 네트워크 포트 상태 확인
ibdev2netdev
```

예상 출력:

```text
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)      <-- 이것을 사용
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
```

> ⚠️ **매우 중요 - 인터페이스 선택 규칙:**

```text
┌─────────────────────────────────────────────────────────────────┐
│  인터페이스 선택 시 주의사항                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 반드시 (Up) 상태인 인터페이스만 사용                         │
│                                                                 │
│  2. 소문자 enp1s0f...로 시작하는 것을 우선 사용                  │
│     ✅ enp1s0f0np0 (소문자 p)                                   │
│     ❌ enP2p1s0f0np0 (대문자 P) - 이것은 무시!                   │
│                                                                 │
│  3. 대문자 P가 포함된 인터페이스(enP2p...)는 사용하지 마세요     │
│     이 인터페이스는 GLOO 통신에서 "Unable to find address"       │
│     에러를 발생시킵니다.                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 IP 주소 확인

```bash
# 활성화된 인터페이스의 IP 주소 확인
ip addr show enp1s0f0np0
```

예상 출력:

```text
3: enp1s0f0np0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP
    link/ether 3c:6d:66:cc:b3:b6 brd ff:ff:ff:ff:ff:ff
    inet 169.254.191.44/16 brd 169.254.255.255 scope link noprefixroute
```

PC2에서도 동일하게 확인:

```bash
ip addr show enp1s0f0np0
# 예: inet 169.254.226.158/16
```

### 3.3 환경 정보 기록

확인한 정보를 기록해두세요:

| 항목 | PC1 (Head) | PC2 (Worker) |
|------|------------|--------------|
| IP 주소 | 169.254.191.44 | 169.254.226.158 |
| 네트워크 인터페이스 | enp1s0f0np0 | enp1s0f0np0 |

> **주의:** 위 IP는 예시입니다. 실제 환경에서 확인한 IP를 사용하세요.

### 3.4 네트워크 연결 테스트

PC1에서 PC2로 ping 테스트:

```bash
ping -c 4 169.254.226.158
```

예상 출력:

```text
PING 169.254.226.158 (169.254.226.158) 56(84) bytes of data.
64 bytes from 169.254.226.158: icmp_seq=1 ttl=64 time=0.123 ms
64 bytes from 169.254.226.158: icmp_seq=2 ttl=64 time=0.098 ms
...
```

## 4. SSH 비밀번호 없이 연결 설정

원활한 클러스터 운영을 위해 SSH 키 기반 인증을 설정합니다.

### 4.1 SSH 키 생성 (PC1에서)

```bash
# SSH 키가 없는 경우 생성
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
```

### 4.2 공개키 복사 (PC1에서 PC2로)

```bash
# PC2로 공개키 복사 (PC2의 비밀번호 입력 필요)
ssh-copy-id tester@169.254.226.158
```

### 4.3 연결 테스트

```bash
# 비밀번호 없이 접속 확인
ssh tester@169.254.226.158 "hostname && echo 'SSH 연결 성공!'"
```

예상 출력:

```text
promaxgb10-957e
SSH 연결 성공!
```

### 4.4 PC2에서 PC1으로도 설정 (선택사항)

PC2에서도 동일하게 설정하면 양방향 SSH가 가능합니다:

```bash
# PC2 터미널에서
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id tester@169.254.191.44
```

## 5. NCCL 빌드 및 테스트

NCCL(NVIDIA Collective Communications Library)은 GPU 간 통신을 최적화합니다. **22GB/s 이상의 대역폭**을 달성해야 405B 모델 추론이 원활합니다.

### 5.1 의존성 설치

PC1과 PC2 모두에서 실행:

```bash
# PC1에서
sudo apt-get update && sudo apt-get install -y libopenmpi-dev

# PC2에서 (SSH 또는 직접)
ssh tester@169.254.226.158 "sudo apt-get update && sudo apt-get install -y libopenmpi-dev"
```

### 5.2 NCCL 소스 클론 및 빌드

PC1에서 실행:

```bash
# NCCL 클론
git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl/
cd ~/nccl/

# Blackwell 아키텍처 지원으로 빌드 (compute_121 = GB10)
make -j$(nproc) src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
```

PC2에서 실행 (SSH로):

```bash
ssh tester@169.254.226.158 "git clone -b v2.28.9-1 https://github.com/NVIDIA/nccl.git ~/nccl/"
ssh tester@169.254.226.158 "cd ~/nccl/ && make -j\$(nproc) src.build NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'"
```

> 빌드에는 시간이 소요됩니다. `"Linking libnccl.so"` 메시지가 나타나면 완료입니다.

### 5.3 NCCL 테스트 빌드

PC1에서 실행:

```bash
# 환경 변수 설정
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

# NCCL 테스트 클론 및 빌드
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
```

PC2에서 실행:

```bash
ssh tester@169.254.226.158 '
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests/
cd ~/nccl-tests/
make MPI=1
'
```

### 5.4 NCCL 통신 테스트 실행

PC1에서 실행:

```bash
# 환경 변수 설정
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

# 네트워크 인터페이스 설정 (실제 인터페이스명으로 변경)
export UCX_NET_DEVICES=enp1s0f0np0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export OMPI_MCA_btl_tcp_if_include=enp1s0f0np0

# 기본 테스트 (32MB 버퍼)
mpirun -np 2 -H 169.254.191.44:1,169.254.226.158:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=$UCX_NET_DEVICES \
  -x NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  $HOME/nccl-tests/build/all_gather_perf
```

### 5.5 대역폭 검증 테스트 (16GB 버퍼)

> 22GB/s 이상이 나와야 합니다!

```bash
mpirun -np 2 -H 169.254.191.44:1,169.254.226.158:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=$UCX_NET_DEVICES \
  -x NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

예상 출력:

```text
#       size         count      type   redop    root     time   algbw   busbw  #wrong
 17179869184    2147483648     float    none      -1   375880   45.71   22.85       0
# Avg bus bandwidth    : 22.855
```

**결과 해석:**

- `busbw` (Bus Bandwidth): 22.85 GB/s - **목표 달성!**
- 이 값이 22GB/s 미만이면 케이블 연결 또는 네트워크 설정을 확인하세요

## 6. vLLM 환경 구성

### 6.1 Docker 이미지 확인 (버전 매우 중요!)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    ⚠️ vLLM 버전 선택 주의사항 ⚠️                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ✅ 반드시 사용해야 하는 버전:                                           │
│     nvcr.io/nvidia/vllm:25.11-py3 (vLLM 0.11.0)                        │
│                                                                         │
│  ❌ 절대 사용하면 안 되는 버전:                                          │
│     nvcr.io/nvidia/vllm:25.12.post1-py3 (vLLM 0.12.0)                  │
│                                                                         │
│  이유:                                                                  │
│  - vLLM 0.12.0에는 KV cache 할당 관련 버그가 있음                       │
│  - 모델 로딩은 성공하나 서버 초기화에서 무한 대기                        │
│  - 405B 모델에서 100% 실패 (7회 이상 테스트 결과)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```bash
# vLLM 이미지가 있는지 확인
docker images | grep vllm
```

이미지가 없으면 Pull:

```bash
docker pull nvcr.io/nvidia/vllm:25.11-py3
```

PC2에서도 동일하게:

```bash
ssh tester@169.254.226.158 "docker pull nvcr.io/nvidia/vllm:25.11-py3"
```

### 6.2 클러스터 실행 스크립트 생성

PC1에서 3개의 스크립트 파일을 생성합니다.

#### 6.2.1 메인 클러스터 스크립트 (run_cluster22.sh)

```bash
cat > ~/run_cluster22.sh << 'EOF'
#!/bin/bash
#
# Launch a Ray cluster inside Docker for vLLM inference (multi-NIC safe).
#

set -uo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
  exit 1
fi

DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"
PATH_TO_HF_HOME="$4"
shift 4

ADDITIONAL_ARGS=("$@")

if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
  echo "Error: Node type must be --head or --worker"
  exit 1
fi

CONTAINER_NAME="node-${RANDOM}"

# Parse VLLM_HOST_IP from -e arguments
VLLM_HOST_IP=""
prev=""
for arg in "${ADDITIONAL_ARGS[@]}"; do
  if [[ "$arg" == "-e" ]]; then
    prev="-e"
    continue
  fi
  if [[ "$prev" == "-e" ]]; then
    if [[ "$arg" == VLLM_HOST_IP=* ]]; then
      VLLM_HOST_IP="${arg#VLLM_HOST_IP=}"
    fi
    prev=""
    continue
  fi
  if [[ "$arg" == -eVLLM_HOST_IP=* ]]; then
    VLLM_HOST_IP="${arg#-eVLLM_HOST_IP=}"
    continue
  fi
  if [[ "$arg" == VLLM_HOST_IP=* ]]; then
    VLLM_HOST_IP="${arg#VLLM_HOST_IP=}"
  fi
done

# Environment variables to force Ray IP in container
RAY_IP_VARS=()
if [ -n "${VLLM_HOST_IP}" ]; then
  RAY_IP_VARS+=(-e "RAY_NODE_IP_ADDRESS=${VLLM_HOST_IP}")
  RAY_IP_VARS+=(-e "RAY_OVERRIDE_NODE_IP_ADDRESS=${VLLM_HOST_IP}")
fi

# Build ray start command
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
  RAY_START_CMD+=" --head --port=6379 --dashboard-host=0.0.0.0"
  if [ -n "${VLLM_HOST_IP}" ]; then
    RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
  fi
else
  RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
  if [ -n "${VLLM_HOST_IP}" ]; then
    RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
  fi
fi

# Launch container
docker run \
  --entrypoint /bin/bash \
  --network host \
  --name "${CONTAINER_NAME}" \
  --shm-size 10.24g \
  --gpus all \
  -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
  "${RAY_IP_VARS[@]}" \
  "${ADDITIONAL_ARGS[@]}" \
  "${DOCKER_IMAGE}" -c "${RAY_START_CMD}; sleep infinity"
EOF
chmod +x ~/run_cluster22.sh
```

#### 6.2.2 Head 노드 스크립트 (ray_master.sh)

> **중요:** IP 주소와 인터페이스명을 본인 환경에 맞게 수정하세요!

```bash
cat > ~/ray_master.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

###############################################
# Configuration - PC1 (Head Node)
# 아래 값들을 본인 환경에 맞게 수정하세요!
###############################################
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.11-py3"
MN_IF_NAME="enp1s0f0np0"          # ibdev2netdev에서 확인한 인터페이스명
HEAD_IP="169.254.191.44"          # PC1의 IP 주소
HF_CACHE="$HOME/.cache/huggingface"

###############################################
# Run Cluster Head Node
###############################################
bash run_cluster22.sh \
  "$VLLM_IMAGE" \
  "$HEAD_IP" \
  --head \
  "$HF_CACHE" \
  -e VLLM_HOST_IP="$HEAD_IP" \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$HEAD_IP" \
  -p 7860:7860
EOF
chmod +x ~/ray_master.sh
```

#### 6.2.3 Worker 노드 스크립트 (ray_worker.sh)

> **중요:** IP 주소와 인터페이스명을 본인 환경에 맞게 수정하세요!

```bash
cat > ~/ray_worker.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

###############################################
# Configuration - PC2 (Worker Node)
# 아래 값들을 본인 환경에 맞게 수정하세요!
###############################################
VLLM_IMAGE="nvcr.io/nvidia/vllm:25.11-py3"
MN_IF_NAME="enp1s0f0np0"          # ibdev2netdev에서 확인한 인터페이스명
HEAD_IP="169.254.191.44"          # PC1(Head)의 IP 주소
WORKER_IP="169.254.226.158"       # PC2(Worker)의 IP 주소
HF_CACHE="$HOME/.cache/huggingface"

###############################################
# Run Worker Node
###############################################
bash run_cluster22.sh \
  "$VLLM_IMAGE" \
  "$HEAD_IP" \
  --worker \
  "$HF_CACHE" \
  -e VLLM_HOST_IP="$WORKER_IP" \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$HEAD_IP"
EOF
chmod +x ~/ray_worker.sh
```

### 6.3 스크립트를 PC2로 복사

```bash
scp ~/run_cluster22.sh ~/ray_worker.sh tester@169.254.226.158:~/
ssh tester@169.254.226.158 "chmod +x ~/run_cluster22.sh ~/ray_worker.sh"
```

### 6.4 HuggingFace 토큰 설정

405B 모델 다운로드를 위해 HuggingFace 토큰이 필요합니다.

1. https://huggingface.co/settings/tokens 에서 토큰 생성
2. 환경 변수로 설정:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

## 7. Ray 클러스터 실행

### 7.1 기존 컨테이너 정리

```bash
# PC1 정리
docker rm -f $(docker ps -a --format '{{.Names}}' | grep -E '^node-') 2>/dev/null

# PC2 정리
ssh tester@169.254.226.158 "docker rm -f \$(docker ps -a --format '{{.Names}}' | grep -E '^node-') 2>/dev/null"
```

### 7.2 Head 노드 시작 (PC1)

터미널 1에서:

```bash
cd ~
bash ray_master.sh
```

예상 출력:

```text
2026-01-22 06:34:16,407  INFO scripts.py:957 -- To add another node to this Ray cluster, run
2026-01-22 06:34:16,407  INFO scripts.py:960 --   ray start --address='169.254.191.44:6379'
...
2026-01-22 06:34:16,408  INFO scripts.py:1122 -- This command will now block forever until terminated by a signal.
```

### 7.3 Worker 노드 시작 (PC2)

새 터미널(터미널 2)에서:

```bash
ssh tester@169.254.226.158 "cd ~ && bash ray_worker.sh"
```

예상 출력:

```text
2026-01-22 06:34:23,534  SUCC scripts.py:1109 -- Ray runtime started.
```

### 7.4 클러스터 상태 확인

새 터미널(터미널 3)에서:

```bash
VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-')
docker exec "$VLLM_CONTAINER" ray status
```

예상 출력:

```text
======== Autoscaler status: 2026-01-22 06:35:55 ========
Node status
---------------------------------------------------------------
Active:
 1 node_xxxxx   <-- PC1
 1 node_yyyyy   <-- PC2
...
Resources
---------------------------------------------------------------
Total Usage:
 0.0/40.0 CPU
 0.0/2.0 GPU     <-- 2개 GPU 사용 가능!
 0B/218.76GiB memory
```

**핵심 확인사항:**

- `Active` 노드가 2개
- GPU가 `2.0` (`0.0/2.0 GPU`)

## 8. 405B 모델 추론 실행

### 8.1 vLLM 서버 시작

터미널 3에서:

```bash
export HF_TOKEN="your_huggingface_token_here"
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-')

docker exec -e HF_TOKEN="$HF_TOKEN" -e HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" "$VLLM_CONTAINER" \
  vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
    --tensor-parallel-size 2 \
    --max-model-len 256 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --kv-cache-dtype fp8
```

**매개변수 설명:**

| 매개변수 | 값 | 설명 |
|----------|----|------|
| `--tensor-parallel-size` | 2 | GPU 2개로 모델 분할 |
| `--max-model-len` | 256 | 최대 시퀀스 길이 |
| `--gpu-memory-utilization` | 0.9 | GPU 메모리 90% 사용 |
| `--enforce-eager` | - | Eager 모드 강제 |
| `--kv-cache-dtype` | fp8 | KV 캐시를 FP8로 저장 (메모리 절약) |

### 8.2 모델 로딩 대기

모델 로딩에는 시간이 걸립니다. 다음 메시지가 나타나면 준비 완료:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 8.3 추론 테스트

새 터미널(터미널 4)에서:

**Health Check**

```bash
curl http://localhost:8000/health
# 응답 없이 HTTP 200이면 정상
```

**Completion API 테스트**

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
    "prompt": "Hello, I am",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

예상 응답:

```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "model": "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
  "choices": [{
    "text": " interested in your spare parts quote...",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 32,
    "total_tokens": 37
  }
}
```

**Chat Completion API 테스트**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## 9. 트러블슈팅 완전 가이드

### 9.1 치명적 에러: vLLM 버전 문제

**증상:** 모델 로딩 후 서버 초기화에서 무한 대기

```text
INFO [gpu_model_runner.py:3549] Model loading took 102.2725 GiB memory and 456.328959 seconds
(이후 아무 로그 없이 멈춤)
```

**원인**

vLLM 0.12.0 (`nvcr.io/nvidia/vllm:25.12.post1-py3`)에 KV cache 할당 버그가 있음

**해결**

```bash
# 잘못된 이미지 삭제
docker rmi nvcr.io/nvidia/vllm:25.12.post1-py3

# 올바른 이미지 사용
docker pull nvcr.io/nvidia/vllm:25.11-py3
```

### 9.2 치명적 에러: 출력이 백슬래시(\)만 나옴

**증상**

```json
{
  "content": "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\..."
}
```

**원인**

- Device Capability 12.1 (GB10)이 SymmMemCommunicator에서 지원되지 않음
- 특정 vLLM 버전에서 발생

**해결**

- vLLM 0.11.0 사용 확인
- `--enforce-eager` 플래그 추가
- `--disable-custom-all-reduce` 시도

```bash
vllm serve ... --enforce-eager --disable-custom-all-reduce
```

### 9.3 "Current node has no GPU available" 에러

**증상**

```text
ValueError: Current node has no GPU available.
current_node_resource={'CPU': 20.0, ...}
```

**원인**

이전 실행에서 생성된 Placement Group이 GPU를 점유하고 있음

**해결**

```bash
# 1. 모든 컨테이너 정리
docker rm -f $(docker ps -a --format '{{.Names}}' | grep -E '^node-') 2>/dev/null
ssh tester@169.254.226.158 "docker rm -f \$(docker ps -a --format '{{.Names}}' | grep -E '^node-') 2>/dev/null"

# 2. Ray 클러스터 다시 시작 (7.2, 7.3 단계)
```

### 9.4 GLOO 인터페이스 에러

**증상**

```text
RuntimeError: Unable to find address for: enP2p1s0f0np0
```

**원인**

대문자 P가 포함된 인터페이스(`enP2p1s0f0np0`)를 사용하려고 함. 이 인터페이스에는 IP가 할당되어 있지 않음.

**해결**

모든 환경변수에서 소문자 인터페이스 사용:

```bash
# 올바른 설정
-e GLOO_SOCKET_IFNAME=enp1s0f0np0    # 소문자 p
-e NCCL_SOCKET_IFNAME=enp1s0f0np0
-e TP_SOCKET_IFNAME=enp1s0f0np0

# 잘못된 설정
-e GLOO_SOCKET_IFNAME=enP2p1s0f0np0  # 대문자 P - 사용 금지!
```

### 9.5 NCCL 통신 에러

**증상**

```text
NCCL error: remote process exited or there was a network error
ncclRemoteError: socketProgress: Connection closed by remote peer
```

**원인**

- 네트워크 인터페이스 설정 불일치
- 방화벽 차단
- Worker 노드 프로세스 비정상 종료

**해결**

```bash
# 1. 양쪽 노드에서 인터페이스 확인
ibdev2netdev

# 2. IP 연결 테스트
ping -c 3 169.254.226.158

# 3. Worker 로그 확인
ssh tester@169.254.226.158 "docker logs \$(docker ps --format '{{.Names}}' | grep -E '^node-') 2>&1 | tail -50"

# 4. 모든 환경변수가 동일한 인터페이스를 사용하는지 확인
# UCX_NET_DEVICES, NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, TP_SOCKET_IFNAME
```

### 9.6 Ray OOM (Out of Memory) 에러

**증상**

```text
ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.
Memory on the node was 113.81GB / 119.64GB (0.951351),
which exceeds the memory usage threshold of 0.95
```

**원인**

70B 이상 대형 모델 로딩 시 시스템 메모리 사용량이 Ray의 기본 임계값(95%)을 초과

**해결**

환경변수로 메모리 임계값 상향:

```bash
# 스크립트에 추가 또는 직접 실행 시 설정
-e RAY_memory_usage_threshold=0.99
```

> 참고: 405B AWQ-INT4 모델은 이 설정 없이도 동작하지만, 70B FP16 모델은 필수

### 9.7 shm-size 부족 에러

**증상**

- 프로세스 간 통신 실패
- 모델 로딩 중 멈춤

**원인**

Docker의 공유 메모리(`/dev/shm`) 크기가 부족

**해결**

```bash
# 올바른 설정 (10GB 이상)
--shm-size 10.24g

# 잘못된 설정
--shm-size 1g  # 너무 작음!
```

### 9.8 NCCL 대역폭이 22GB/s 미만

**증상**

```text
# Avg bus bandwidth    : 15.xxx  (22 미만)
```

**원인**

- QSFP 케이블 연결 불량
- 잘못된 인터페이스 사용
- 환경변수 미설정

**해결**

```bash
# 1. 케이블 재연결
# 물리적으로 QSFP 케이블을 분리했다가 다시 연결

# 2. 인터페이스 상태 확인
ibdev2netdev
# (Up) 상태인 enp1s0f... 인터페이스 사용

# 3. 환경변수 확인
echo $UCX_NET_DEVICES
echo $NCCL_SOCKET_IFNAME
echo $OMPI_MCA_btl_tcp_if_include
```

### 9.9 Ray 노드가 1개만 보임

**증상**

```text
Active:
 1 node_xxxxx   <-- PC1만 보임
```

**원인**

- Worker 스크립트가 실행되지 않음
- Head IP 주소 불일치
- 네트워크 연결 문제

**해결**

```bash
# 1. PC2에서 컨테이너 실행 확인
ssh tester@169.254.226.158 "docker ps"

# 2. Worker 로그 확인
ssh tester@169.254.226.158 "docker logs \$(docker ps --format '{{.Names}}' | grep -E '^node-') 2>&1 | tail -30"

# 3. ray_worker.sh의 HEAD_IP가 PC1의 IP와 일치하는지 확인
cat ~/ray_worker.sh | grep HEAD_IP
```

### 9.10 Docker 권한 문제

**증상**

```text
permission denied while trying to connect to the Docker daemon socket
```

**해결**

```bash
# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 로그아웃 후 다시 로그인
exit
# (다시 로그인)

# 또는 일시적으로 sudo 사용
sudo docker ...
```

### 9.11 모델 파일 권한 문제

**증상**

```text
PermissionError: [Errno 13] Permission denied: '/root/.cache/huggingface/...'
```

**해결**

```bash
# 캐시 디렉토리 소유권 변경
sudo chown -R $USER:$USER ~/.cache/huggingface

# PC2에서도 동일하게
ssh tester@169.254.226.158 "sudo chown -R \$USER:\$USER ~/.cache/huggingface"
```

### 9.12 Device Capability 12.1 경고

**증상**

```text
WARNING [symm_mem.py:67] SymmMemCommunicator: Device capability 12.1 not supported
WARNING [custom_all_reduce.py:92] Custom allreduce is disabled because this process group spans across nodes
```

**설명**

이 경고는 무시해도 됩니다. GB10 (Blackwell) GPU의 SM_121a 아키텍처가 일부 최적화 기능에서 아직 완전히 지원되지 않아 발생하는 경고입니다. 기본 NCCL 통신은 정상 작동합니다.

### 9.13 V1 Engine 초기화 타임아웃

**증상**

```text
KeyboardInterrupt: (600초 이상 대기 후 강제 종료)
```

**원인**

- vLLM 0.12.0 버전 사용
- GPU 리소스 점유
- 메모리 부족

**해결**

- vLLM 0.11.0 사용 확인
- 모든 컨테이너 정리 후 재시작
- `--gpu-memory-utilization` 값 낮추기 (0.85 → 0.80)

## 10. 정리 및 종료

### 10.1 vLLM 서버 종료

모델 서버가 실행 중인 터미널에서 `Ctrl+C`

### 10.2 모든 컨테이너 정리

```bash
# PC1
docker stop $(docker ps --format '{{.Names}}' | grep -E '^node-') 2>/dev/null
docker rm $(docker ps -a --format '{{.Names}}' | grep -E '^node-') 2>/dev/null

# PC2
ssh tester@169.254.226.158 "docker stop \$(docker ps --format '{{.Names}}' | grep -E '^node-') 2>/dev/null"
ssh tester@169.254.226.158 "docker rm \$(docker ps -a --format '{{.Names}}' | grep -E '^node-') 2>/dev/null"
```

### 10.3 NCCL 빌드 정리 (선택사항)

더 이상 NCCL 테스트가 필요 없다면:

```bash
rm -rf ~/nccl/
rm -rf ~/nccl-tests/

ssh tester@169.254.226.158 "rm -rf ~/nccl/ ~/nccl-tests/"
```

## 빠른 참조 카드

핵심 명령어 모음:

```bash
# === 환경 정보 확인 ===
ibdev2netdev                    # 네트워크 인터페이스 상태
ip addr show enp1s0f0np0        # IP 주소 확인

# === 클러스터 시작 ===
cd ~ && bash ray_master.sh      # PC1: Head 노드 시작
ssh PC2 "bash ray_worker.sh"    # PC2: Worker 노드 시작

# === 상태 확인 ===
VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-')
docker exec "$VLLM_CONTAINER" ray status

# === 405B 모델 시작 ===
docker exec -e HF_TOKEN="$HF_TOKEN" "$VLLM_CONTAINER" \
  vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
    --tensor-parallel-size 2 --max-model-len 256 \
    --gpu-memory-utilization 0.9 --enforce-eager --kv-cache-dtype fp8

# === 추론 테스트 ===
curl http://localhost:8000/health
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4","prompt":"Hello","max_tokens":32}'

# === 정리 ===
docker rm -f $(docker ps -a --format '{{.Names}}' | grep -E '^node-')
```

## 부록 A: 성공적인 테스트 결과 예시

### NCCL 테스트 결과

```text
#       size         count    busbw
 17179869184    2147483648    22.86 GB/s  <-- 목표 달성!
```

### Ray 클러스터 상태

- Active: 2 nodes
- GPU: 0.0/2.0 (available)
- Memory: 218.76 GiB

### 405B 추론 성능

- Model: Meta-Llama-3.1-405B-Instruct-AWQ-INT4
- Prompt tokens: 5
- Completion tokens: 32
- Response time: ~20 seconds

## 부록 B: 실패한 설정들 (참고용)

> 아래 설정들은 모두 실패했습니다. 사용하지 마세요.

### B.1 실패한 vLLM 버전

```bash
# ❌ 사용 금지
nvcr.io/nvidia/vllm:25.12.post1-py3  # KV cache 할당 버그
```

### B.2 실패한 shm-size

```bash
# ❌ 사용 금지
--shm-size 1g  # 프로세스 간 통신 실패
```

### B.3 실패한 인터페이스

```bash
# ❌ 사용 금지
-e GLOO_SOCKET_IFNAME=enP2p1s0f0np0  # 대문자 P - IP 없음
```

### B.4 실패한 메모리 설정 (405B with vLLM 0.12)

```bash
# ❌ vLLM 0.12.0에서는 어떤 설정으로도 실패
--gpu-memory-utilization 0.95  # 시작 체크 실패
--gpu-memory-utilization 0.90  # 드라이버 OOM
--gpu-memory-utilization 0.87  # KV cache 할당에서 멈춤
--max-model-len 1              # 극단적 최소값도 실패
```

## 부록 C: 환경 변수 전체 목록

| 변수 | 값 | 용도 |
|------|----|------|
| `VLLM_HOST_IP` | (각 노드 IP) | vLLM 호스트 IP |
| `RAY_NODE_IP_ADDRESS` | (각 노드 IP) | Ray 노드 IP |
| `RAY_OVERRIDE_NODE_IP_ADDRESS` | (각 노드 IP) | Ray IP 오버라이드 |
| `UCX_NET_DEVICES` | enp1s0f0np0 | UCX 네트워크 디바이스 |
| `NCCL_SOCKET_IFNAME` | enp1s0f0np0 | NCCL 소켓 인터페이스 |
| `OMPI_MCA_btl_tcp_if_include` | enp1s0f0np0 | OpenMPI TCP 인터페이스 |
| `GLOO_SOCKET_IFNAME` | enp1s0f0np0 | GLOO 소켓 인터페이스 |
| `TP_SOCKET_IFNAME` | enp1s0f0np0 | Tensor Parallel 인터페이스 |
| `MASTER_ADDR` | (Head IP) | 마스터 노드 주소 |
| `RAY_memory_monitor_refresh_ms` | 0 | Ray 메모리 모니터 비활성화 |

---

**문서 버전:** 2.0 | **작성일:** 2026-01-22 | **테스트 환경:** Dell Pro Max with GB10 (GB10 GPU) x 2, Ubuntu 24.04, vLLM 0.11.0

**변경 이력:**

- **v1.0** (2026-01-22): 초기 버전
- **v2.0** (2026-01-22): 트러블슈팅 완전 가이드 추가, 실패 사례 문서화
