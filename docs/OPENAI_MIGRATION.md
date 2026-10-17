# OpenAI 리포트 모델 변경

| 항목 | 기존 | 정리본 |
|---|---|---|
| 사진·압력 등 리포트 문구 | gpt-4.1-mini | gpt-6-astra |
| API | Responses | Responses 유지 |
| 출력 | strict JSON Schema | strict JSON Schema + 서버 JSON Schema 검증 |
| reasoning | 생략 | low 명시 |
| 출력 토큰 | 호출별 작은 한도 | 추론 모델 최소 8,000, 재시도 상한 16,000 |
| 오류 처리 | 제한 재시도와 로컬 대체 문구 | 유지, 형식 오류도 검증 후 처리 |
| 피부 평가 범위 | 발등·발바닥 | 양발 발등만 |

숫자와 마스크는 로컬 모델이 계산합니다. GPT는 점수를 재계산하지 않고 설명을 생성합니다.
무좀·피부반응 설명에는 발바닥 판단을 포함하지 않습니다. 발바닥 사진은 아치·눌림 설명에 사용합니다.
요청은 `store=false`, 엄격한 스키마, 제한된 총 시도 횟수와 전체 시간 한도를 사용합니다.
시간 초과, 잘린 출력, 거절, API 오류, 잘못된 JSON은 처리하고 계산된 사실을 바탕으로 대체 문구를 만듭니다.
대체 문구는 `DETERMINISTIC_FALLBACK`과 오류 코드로 구분합니다. 의료 정확도 향상이나 무오류를 보장하지 않습니다.

현재 계정의 모델 조회 API에서 gpt-6-astra 접근 가능 여부를 확인했습니다.
환자 사진 없이 합성 입력으로 실제 Responses 요청을 보내 지정된 JSON Schema의 응답을 확인했습니다.
레거시 비활성 pressed-region 모델 설정과 선택 기능의 로컬 Ollama 모델은 변경하지 않았습니다.
모델 계정 권한·비용·응답 속도는 실제 배포 환경에서 확인해야 합니다.

공식 문서:
- [GPT-6 사용 및 마이그레이션](https://developers.openai.com/api/docs/guides/latest-model)
- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
