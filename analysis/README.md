# 중간 보고서 재분석

`logs/main`, `logs/pilot`의 원본 JSONL 로그에서 중간 보고서(2026년 10월)의 모든 수치와 그림 1–5를 다시 만든다. 하네스 집계는 표 4(오류 추가 손실)와 상호작용 검정에만 쓰며(`logs/main/summary_penalty.csv`, `summary_interaction.csv`), 점추정은 로그에서 다시 계산해 대조한다.

## 실행

```bash
pip install -r analysis/requirements.txt
python analysis/reanalyze.py --repo . --out analysis/out
```

- 표 2의 조건부 로짓 부트스트랩(2,000회) 때문에 약 10분 걸린다. 그림만 빠르게 확인하려면 `--logit-boot 0`을 붙인다(이때 표 2 구간과 그림 4 구간 값은 보고서와 달라진다).
- 그림의 한글은 Noto Sans CJK 글꼴이 설치돼 있어야 제대로 보인다.
- 모든 무작위 재표집은 seed 20261018로 고정되어 있어 같은 로그에서 같은 `numbers.json`이 나온다.

## 출력 (`analysis/out/`, git 제외)

| 파일 | 내용 |
|---|---|
| `numbers.json` | 보고서 본문·표·부록에 쓰인 모든 값 |
| `fig1_success.pdf` | 조건·horizon별 성공률 |
| `fig2_position.pdf` | free h1 위치 선호 |
| `fig3_format.pdf` | 형식 붕괴와 지속성 |
| `fig4_net.pdf` | FP16 대비 성공률 차이 |
| `fig5_efficiency.pdf` | 효율 지표 |

## 주요 정의

- **시나리오 클러스터**: 복구 과제의 `recovery_*_h1/_h3` 기본 ID에서 접두어·접미어를 떼어 free 과제와 같은 60개 시나리오로 묶는다.
- **표 3**: 첫 형식 오류 이전에 `inspect_state`를 의도한 응답만 센다. 직전 메시지가 모델이 실행한 확정 결과인 경우와 복구 조건에 주입된 확정 결과인 경우를 따로 집계한다.
- **형식 오류의 성공률 영향**: 첫 형식 오류 이전에 `inspect_state`가 정상 실행된 직후 응답의 수리 비율을, 형식 오류가 난 실패 run 수에 곱한 근사값이다(선택 편향 있음).
- **확정 직후 행동**: 모델이 실행한 확정의 결과 직후 모든 응답을 의도 도구·완료 선언·형식 오류로 분류한다.
- **응답 시퀀스 동일**: 각 응답의 행동 유형·도구 이름·인자(형식 오류는 오류 종류)가 처음부터 끝까지 같은 짝의 비율이다.
