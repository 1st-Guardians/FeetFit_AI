# 가중치와 카메라 보정 파일

`manifest.json`에 있는 가중치 4개와 카메라 보정 파일 1개, 총 5개 파일을 이 폴더에 넣습니다. 파일명은 그대로 유지합니다.
실행에 사용하는 모델은 YOLO, SegFormer-B2 Teacher–Student 무좀 모델, SegFormer 상처 모델, SegFormer 발바닥 눌림 모델입니다.
카메라 보정 파일도 같은 폴더에 둡니다. 무지외반은 Contour HVA v6이므로 별도 학습 가중치가 필요 없습니다.

바이너리 파일은 Git에서 제외됩니다. 이 README와 체크섬 목록만 업로드합니다.
설치 확인: `python scripts/check_setup.py --verify-sha256`

다른 카메라를 쓰면 해당 카메라로 보정한 NPZ가 필요합니다. 현재 보정 파일은 기존 1280×720 장치용입니다.
신발 추천용 BGE-M3와 Ollama는 선택 서비스이며 별도 다운로드 또는 실행이 필요합니다.

무좀 가중치: `tinea_segformer_b2_teacher_student_best.pt` (약 104 MiB). 최종 Student의 무좀 1채널 가중치, RGB 384 입력 설정과 검증 평균 Dice 기준 최적 임계값 **0.30**이 함께 들어 있습니다. 추론은 체크포인트 임계값을 사용하며 학습 폴더나 온라인 다운로드를 요구하지 않습니다. 기존 `tinea_pedis_best.pt`는 새 무좀 추론에서 읽지 않습니다. `sam_vit_b_01ec64.pth`는 현재 API의 분석 경로에서 사용하지 않으며 설치 검사 대상에서도 제외합니다. 레거시 SAM 코드는 남아 있지만 기본 분석에서 실행하지 않습니다.

현재 분석에 필요한 파일은 아래 목록이 전부입니다.

| 파일 | 용도 |
| --- | --- |
| `foot_seg_yolo11n_best.pt` | 발 영역 검출 |
| `tinea_segformer_b2_teacher_student_best.pt` | 발등 무좀 |
| `wound_segformer_experiment_c_best.pt` | 발등 상처·피부반응 |
| `plantar_contact_segformer_b0_real_finetune_brightness2_best.pt` | 발바닥 눌림 |
| `camera_calibration.npz` | 카메라 보정 (모델 가중치 아님) |

기존 가중치 파일을 삭제하거나 이름만 바꾸지 않아도 됩니다. 위 목록 외의 로컬 가중치는 현재 기본 분석에서 읽지 않습니다.
