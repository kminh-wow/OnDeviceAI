# Quantization × Long-horizon Agent 실험 코드

Qwen3-1.7B를 FP16 / Q8_0 / Q4_K_M로 직접 PTQ한 GGUF를 llama.cpp로 돌려서,
gold 호출 1 / 3 / 6개 Agent 과제에서 성공률·신뢰성·효율을 비교한다.

## 폴더 배치

```
C:\Users\kminh\
├─ llama-bin\llama-server.exe        (CPU 빌드, b11338)
├─ llama-bin-vk\llama-server.exe     (Vulkan 빌드, b11338)
├─ qwen3-1.7b-f16.gguf, qwen3-1.7b-q8_0.gguf, qwen3-1.7b-q4_k_m.gguf
└─ agent_quant\                      ← 여기서 명령 실행
```

## 실행 순서

```powershell
pip install -r requirements.txt
python tasks.py --lang en                          # 120문제 생성 + 정답 유일성 검사
python run_experiment.py --oracle --out results_oracle   # 100% 나와야 정상

# 본 실험 (task v2, 영어, thinking OFF, 3회 반복)
python run_experiment.py --models ..\qwen3-1.7b-f16.gguf ..\qwen3-1.7b-q8_0.gguf ..\qwen3-1.7b-q4_k_m.gguf --repeats 3 --mmap --out results_v2
python analyze.py --results results_v2

# 채점 신뢰도 검증 (사람 채점)
python audit.py sample --results results_v2 --n 40      # audit_sheet.csv 채우기
python audit.py score --results results_v2

# peak RAM 재측정 (mmap 끔, 정밀도별 3문제)
python run_experiment.py --models ..\qwen3-1.7b-f16.gguf ..\qwen3-1.7b-q8_0.gguf ..\qwen3-1.7b-q4_k_m.gguf --limit 1 --load-mode none --out results_ram
```

- 끊겨도 같은 명령을 다시 실행하면 이어서 한다. 다른 task 버전 결과가 있는 폴더에는 이어 쓰지 않는다(`task_version` 검사).
- `--load-mode none`: llama.cpp b11338에서 mmap을 끄는 옵션(구 `--no-mmap`). `--mmap`은 `--load-mode auto`와 같다.
- `--ngl 99 --server ..\llama-bin-vk\llama-server.exe`: 내장 GPU(Vulkan) 사용.

## 과제 (task v2: 4개 영역 × 10 인스턴스 × 3 길이 = 120문제)

| 영역 | 1 call | 3 calls | 6 calls |
|---|---|---|---|
| schedule | 첫 일정 시작/마지막 일정 종료 | 일정→식당 검색→예약 | 일정→날씨→(비면 실내) 식당→예산→예약→일정 등록 |
| travel | 경로 시간(대중교통/택시) | 박물관 검색→경로→일정 등록 | 날씨→장소→대중교통·택시 경로→예산→이동수단 포함 일정 등록 |
| place | 날씨·기온 | 일정→날씨→(비면 실내) 카페 추천 | 두 날짜 날씨 비교→일정→장소→경로→일정 등록 |
| shopping | 최저가/최고 평점 가격 | 예산→검색→장바구니 | 예산→2종 검색→순차 담기→장바구니 확인 |

정답 유일성 (`tasks.check_tasks`): 평점·가격 동점이면 생성 단계에서 오류, 모든 예산은 `X000 + 500`원이라 가격 합계와 같아질 수 없음("이하/미만" 모호성 제거). 비 오는 날 실내 장소 선택은 `indoor=True`로 검색하든 전체 검색 후 직접 고르든 정답으로 인정.

## 지표

| 지표 | 정의 |
|---|---|
| success | 필요한 호출 전부 일치 + 최종 상태(예약·일정·장바구니) 일치 + (조회형) 답변에 핵심 값 포함 |
| step_accuracy / partial_match | gold 호출 중 정확히 재현한 비율 (NESTFUL partial) |
| full_match | gold 호출 전부 + 잘못된 추가 호출 없음 (NESTFUL full) |
| f1_func, f1_param | 함수 이름 / (도구, 인자, 값) F1 (Granite, NESTFUL) |
| first_error_type | 첫 이탈 유형 (아래) / first_hard_error_type: extra_lookup 제외 |
| progress | 첫 hard error 전까지 재현한 gold 호출 수 |
| recovered | hard error 후 성공 (extra_lookup, over_execution 제외) |

오류 유형: `format` · `tool_selection`(없는 도구, 불필요한 쓰기 도구) · `extra_lookup`(불필요한 조회 도구) ·
`argument` · `observation` · `skipped_step` · `over_execution` · `premature_stop` · `max_steps` · `state` · `final_answer`
(뒤의 네 개는 n_calls+1 위치로 기록되어 상대 위치 분석에서 제외)

## analyze.py 출력

`summary_by_length.csv`(길이별 성공률 + 과제 단위 bootstrap 95% CI), `summary_overall.csv`(gold 호출 수 기준 감소 기울기, 효율),
`summary_by_derived.csv`(이전 결과에서 가져와야 하는 인자 수별), `stats_logit.csv`(정밀도×길이 로지스틱 회귀, 과제 클러스터 SE, 상호작용 Wald 검정),
`stats_pairwise.csv`(정밀도 쌍별 McNemar + 차이의 paired bootstrap CI), `compounding.csv`(p^T 예측 vs 실제),
`error_types.csv` / `error_types_hard.csv`, `first_error_position.csv`(상대 위치), `progress_curve.csv`, `recovery.csv`(분모 포함), `analysis.txt`(전체 로그)

## 외부 검증: BFCL multi-turn

주 실험(직접 설계한 60문제)과 별개로, BFCL의 multi-turn 데이터(`multi_turn_base`, BFCL V3 multi-turn과 같은 200문제)로 같은 3개 정밀도를 평가한다.
BFCL 공식 가상 API 코드(`bfcl/func_source_code`, Apache-2.0)를 그대로 쓰고, 채점도 BFCL과 같은 방식(매 턴 종료 후 상태 비교 + 실행 결과 포함 여부)이다.

BFCL은 step 수를 미리 고정할 수 없어서, **정답 호출 수로 사후 구간화**한다: short 2~4 / medium 5~6 / long 7+ (구간마다 같은 문제를 층화 추출, 기본 15개씩 = 45문제).

```powershell
pip install mpmath
python bfcl_runner.py --oracle                                               # 100% 나와야 정상
python bfcl_runner.py --models ..\qwen3-1.7b-q4_k_m.gguf --per-bin 2 --mmap --out bfcl_smoke
python bfcl_runner.py --models ..\qwen3-1.7b-f16.gguf ..\qwen3-1.7b-q8_0.gguf ..\qwen3-1.7b-q4_k_m.gguf --mmap
python bfcl_analyze.py
```

| 출력 | 의미 |
|---|---|
| `accuracy` | BFCL 공식 지표와 같은 multi-turn 정확도 (모든 턴 통과) |
| `turns_passed` | 첫 실패 전까지 통과한 턴 비율 (오류가 얼마나 일찍 시작되는지) |
| `first_fail_reason` | `empty_turn`(도구 호출 안 함) / `state_mismatch`(상태 틀림) / `response_mismatch`(필요한 조회 결과 누락) |
| `error_call_rate` | 실행 중 에러를 낸 호출 비율 |

주의: BFCL 문서가 길어서 컨텍스트를 16K로 쓴다(`--ctx`). 주 실험이 돌아가는 동안에는 같이 돌리지 말 것(포트·CPU 충돌).
