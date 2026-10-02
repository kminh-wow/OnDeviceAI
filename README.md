# Quantization × Long-horizon Agent 실험 코드

Qwen3-1.7B를 FP16 / Q8_0 / Q4_K_M로 직접 PTQ한 GGUF를 llama.cpp로 돌려서,
1 / 3 / 5+ step Agent 과제에서 성공률·신뢰성·효율을 비교한다.

## 폴더 배치

```
C:\Users\kminh\
├─ llama-bin\llama-server.exe
├─ qwen3-1.7b-f16.gguf, qwen3-1.7b-q8_0.gguf, qwen3-1.7b-q4_k_m.gguf
└─ agent_quant\        ← 이 폴더 (여기서 명령 실행)
```

## 실행 순서

```powershell
cd C:\Users\kminh\agent_quant
pip install -r requirements.txt

# 0) 문제 60개 확인 (tasks_ko.json 생성)
python tasks.py --lang ko

# 1) 파이프라인 검증: 정답 궤적을 그대로 재생 → 100% 나와야 정상
python run_experiment.py --oracle --out results_oracle

# 2) 스모크 테스트: Q4로 horizon별 2문제씩 (시간 측정용)
python run_experiment.py --models ..\qwen3-1.7b-q4_k_m.gguf --limit 2 --out results_smoke

# 3) 본 실험: 3개 정밀도 × 60문제 × 3회 반복
python run_experiment.py --models ..\qwen3-1.7b-f16.gguf ..\qwen3-1.7b-q8_0.gguf ..\qwen3-1.7b-q4_k_m.gguf --repeats 3

# 4) 분석 (표 + 그래프)
python analyze.py
```

- 중간에 끊겨도 같은 명령을 다시 실행하면 끝난 것은 건너뛰고 이어서 한다.
- 영어 버전: `--lang en` (한국어 이해력 효과를 분리하고 싶을 때)
- Qwen3 thinking 모드: 기본 OFF, 켜려면 `--think` (CPU에서는 매우 느려짐)

## 파일

| 파일 | 역할 |
|---|---|
| `tools_env.py` | Mock Tool 10개 + 결정적(deterministic) 가상 세계 |
| `tasks.py` | 4개 영역 × 5문제 × 3 horizon = 60문제, 정답 궤적·최종 상태 자동 생성 |
| `agent.py` | llama-server(OpenAI 호환 API) 위의 tool-calling 루프 |
| `evaluate.py` | 성공 여부, step accuracy, 첫 오류 위치·유형 판정 |
| `run_experiment.py` | 모델별 서버 자동 실행, 전체 과제 수행, peak RAM 측정 |
| `analyze.py` | 요약 표(csv)와 그래프(png) 생성 |

## 과제 구성

| 영역 | 1-step | 3-step | 5+ step (6 calls) |
|---|---|---|---|
| schedule (일정·예약) | 일정 조회 | 일정→식당 검색→예약 | 일정→날씨→(비면 실내) 식당→예산→예약→캘린더 등록 |
| travel (여행·경로) | 경로 시간 | 박물관 검색→경로→일정 등록 | 날씨→장소 선택→대중교통·택시 경로→예산→이동수단 포함 일정 등록 |
| place (장소·날씨) | 날씨 조회 | 일정→날씨→(비면 실내) 카페 추천 | 두 날짜 날씨 비교→일정→장소→경로→빈 시간에 일정 등록 |
| shopping (쇼핑·예산) | 최저가 조회 | 예산→상품 검색→장바구니 | 예산→상품 2종 검색→순차 담기(남은 예산 반영)→장바구니 확인 |

## 슬라이드 지표 ↔ 출력 컬럼

| 슬라이드 | 컬럼 (`summary_by_horizon.csv`, `summary_overall.csv`) |
|---|---|
| Task Success Rate | `success_mean`, `success_std` (반복 간 표준편차) |
| Step Accuracy | `step_accuracy` (정답 호출 중 맞힌 비율) |
| Valid Tool Call | `valid_call_rate` (형식·스키마가 유효한 호출 비율) |
| Success Decay | `decay_slope_per_step` (horizon에 대한 성공률 기울기) |
| First-error Step | `first_error_step` (오류 난 run에서 첫 오류가 몇 번째 호출인지) |
| 오류 후 회복률 | `recovery_rate` (오류가 있었지만 결국 성공한 비율) |
| Model Size / Peak RAM | `size_mb`, `peak_ram_mb` (`model_meta.json`) |
| Latency | `task_latency_s`, `gen_tps` |

첫 오류 유형 (`error_types.csv`, `figures/error_types.png`):
`format`(호출 형식 깨짐) · `tool_selection`(도구 선택) · `argument`(질문에서 가져올 인자 오류) ·
`observation`(이전 도구 결과에서 가져올 인자 오류 = 상태 추적) · `premature_stop` / `max_steps`(종료 판단) ·
`final_answer`(최종 응답 오류)

## 메모

- LangGraph 대신 같은 역할(상태·최대 step·종료 조건만 통제, 경로는 강제하지 않음)의 루프를 직접 구현했다. 의존성이 적어 재현이 쉽다.
- 정답 판정은 순서를 강제하지 않는다. 필요한 호출을 모두 맞게 하고, 최종 상태(예약·일정·장바구니)가 정확하고, 조회형 과제는 답변에 핵심 값이 들어 있어야 성공.
- 같은 정답 호출을 한 번 더 하는 것은 오류로 보지 않는다(`redundant_calls`). 필요 없는 도구 호출은 `tool_selection` 오류로 센다.
- RAM은 `--no-mmap`으로 모델을 실제로 메모리에 올린 상태에서 llama-server 프로세스의 peak를 잰다.
