# 현재 모델 구성 검증 기록

검증일: 2026-10-18. Windows / Python 3.10.20 / PyTorch 2.5.1+cu121 환경에서 확인했습니다.

현재 발 분석의 활성 모델 가중치는 **4개**이며, 별도로 카메라 보정 파일 **1개**가 필요합니다.

| 모델 | 가중치 |
| --- | --- |
| 발 영역 검출 | `foot_seg_yolo11n_best.pt` |
| 발등 무좀 | `tinea_segformer_b2_teacher_student_best.pt` |
| 발등 상처·피부반응 | `wound_segformer_experiment_c_best.pt` |
| 발바닥 눌림 | `plantar_contact_segformer_b0_real_finetune_brightness2_best.pt` |

카메라 보정은 `camera_calibration.npz`를 사용합니다. 설치 목록과 체크섬 목록에서도 미사용 SAM 가중치를 제외했습니다. 레거시 코드와 보관된 가중치 파일을 삭제하지는 않았습니다.

## 확인한 내용

- 전체 단위·API 계약 테스트 **308개 통과**. 발등만 점수 집계, 발바닥 피부 추론 차단, 무좀·상처 중첩 처리, 압력 회차 전달, GPT 출력 검증·재시도 및 대체 문구를 포함합니다.
- `scripts/check_setup.py --verify-sha256 --load-models --device cpu` 성공. 필수 파일 5개의 체크섬, 활성 모델 4개 및 보정 파일 로드를 확인했습니다.
- 별도 실행에서는 SAM 가중치 경로를 존재하지 않는 경로로 대체한 뒤 실제 `torch.load` 호출을 감시했습니다. 읽은 체크포인트는 위 4개뿐이었습니다.
- 활성 모델 4개 각각 CPU에서 합성 이미지 추론을 실행했고 출력 크기와 유한한 값을 확인했습니다. 무좀 모델은 실제 `SegformerForSemanticSegmentation`으로 로드됐습니다.
- 신규 설치 검사 회귀 테스트는 가중치 바이너리가 없는 Git clone에서도 실행되며, 누락된 레거시 SAM 경로가 필수 검사에 포함되지 않는지 확인합니다.
- 모델·보정 바이너리, 촬영 사진, `.env`는 Git에서 제외하고 소스·문서·체크섬 목록만 커밋합니다.

이는 설치·모델 로드·추론 경로 및 API 계약 검증입니다. 실제 하드웨어와 운영 백엔드 전체를 연결한 검증이나 합성 이미지의 의학적 판정 정확도 검증을 뜻하지 않습니다. 학습 실험의 Dice/IoU와 그 한계는 [무좀 모델 문서](TINEA_SEGFORMER_STUDENT.md)에 별도로 기록되어 있습니다.

초기 패키징 때는 임시 8017번 서버의 `/health`와 `/openapi.json` 및 합성 입력의 실제 OpenAI Structured Outputs 응답을 확인했습니다. 현재 실행에는 배포 환경의 백엔드 주소와 OpenAI API 키가 필요합니다.
