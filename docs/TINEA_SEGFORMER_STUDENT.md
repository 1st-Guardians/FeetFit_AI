# 무좀 모델: SegFormer-B2 Teacher–Student

발등 무좀 추론은 최종 Student 가중치 `weights/tinea_segformer_b2_teacher_student_best.pt`를 사용합니다. 무좀만 학습한 1채널 모델이며 배경 및 기존 염증 클래스는 무좀 정답에 포함되지 않았습니다. 서버에서는 Student 한 개로 추론합니다.

## 추론 설정

- 기존 발 검출 마스크의 경계 상자를 여백 없이 잘라 RGB 384×384로 PIL bilinear resize합니다. 색 보정이나 letterbox를 추가하지 않습니다.
- 픽셀을 255로 나눈 뒤 ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`로 정규화합니다.
- decoder logit을 384×384로 bilinear 복원하고 sigmoid를 적용한 뒤 확률을 발 crop 원본 크기로 복원합니다. 전체 사진 좌표로 되돌리고 발 마스크 밖 확률은 0으로 만듭니다.
- 무좀 마스크는 **확률 ≥ 0.30**입니다. 검증 84장의 평균 Dice가 가장 높은 임계값이며 테스트 데이터로 선택하지 않았습니다. 체크포인트 값이 우선하므로 이전 `FUNGAL_THRESHOLD` 환경변수는 무시합니다.
- 기존 작은 영역 필터, 검출 기준, 상처와 겹칠 때의 처리 및 파란색 시각화는 유지합니다. 최종 표시 마스크에는 이 규칙도 적용됩니다. 안전 점수는 아래 면적 기반 정책을 사용합니다.
- 무좀 모델의 염증 채널은 없으며 상처·피부반응은 기존 독립 모델에서 받습니다. 발바닥 피부 병변 추론은 활성화하지 않습니다.

기존 `TINEA_SLIDING_WINDOW_*`, `TINEA_PREPROCESS_*`는 SAM용 설정이며 이 모델에서는 사용하지 않습니다. 응답 metrics의 `fungal_model`, `fungal_threshold`, `fungal_threshold_source`, `fungal_input_roi_xyxy`, `fungal_preprocessing_policy`에서 실제 적용값을 확인할 수 있습니다.

## 무좀·피부반응 안전 점수

겹침 배정과 검출 필터를 통과한 최종 클래스 마스크로 발 면적 대비 의심 영역 비율 `r`을 계산합니다. 최종 의심 영역이 없으면 100점, 있으면 `65 * exp(-7 * r)`로 계산합니다. 예를 들어 발 면적의 1%는 60.61점, 5%는 45.80점, 10%는 32.28점입니다. 예측 확률과 발 밖의 영역은 표시 점수의 면적 비율에 추가 감점을 만들지 않습니다.

무좀의 기존 신호 검출은 면적·확률 부담으로 계산하는 별도 `detection_score`와 기존 95점 기준을 유지합니다. 표시 점수를 낮췄다는 이유로 이전에 제외했던 후보를 새로 검출하지 않습니다. 독립 피부반응 모델은 최종 마스크에 픽셀이 남아 있으면 검출로 처리합니다.

좌우 종합 점수는 항목별로 더 낮은 안전 점수를 사용합니다. 한쪽이 65점이고 다른 쪽이 100점이면 최종 65점으로 저장하며, API의 정수 점수는 반올림합니다. 의심 영역 지도, 사진 오버레이, 점수 계산은 동일한 최종 마스크를 사용합니다.

## 학습 결과와 서버 결과의 구분

정답 학습 615장, 검증 84장, 테스트 75장으로 분리했고 비라벨 후보 1,352장에서 신뢰도 기준으로 선택한 의사 라벨을 3회 학습에 사용했습니다. 마지막 라운드의 epoch 10 Student를 선택했습니다.

| 평가 | 평균 Dice | 평균 IoU | 전체 픽셀 Dice | 전체 픽셀 IoU |
| --- | ---: | ---: | ---: | ---: |
| 검증 84장, 임계값 0.30 | 0.6756 | 0.5536 | 0.7327 | 0.5781 |
| 테스트 75장, 임계값 0.30 | 0.6035 | 0.4903 | 0.6945 | 0.5319 |

이 수치는 무좀 전경만 평가한 학습 실험의 384×384 정답 발 crop 결과입니다. 서버의 발 검출·영역 필터·상처 중첩 처리까지 포함한 새 평가 수치는 아닙니다. 환자 식별자가 없어 환자 단위 독립성은 확인하지 못했습니다.

## 이식 및 확인

추론에 학습 머신의 경로, 별도 SAM backbone, 인터넷 연결은 필요하지 않습니다. Transformers 4.x 학습 가중치 이름을 설치된 5.x 이름으로 변환한 후 모든 키를 `strict=True`로 로드합니다. 수치 검증에서는 원래 학습 구현과 같은 이미지를 사용합니다.

가중치를 다시 내보낼 때는 학습 결과의 `best.pt`를 사용합니다. 출력 파일은 Git에서 제외됩니다.

```powershell
python scripts/export_fungal_student.py --source "학습결과/best.pt" --output weights/tinea_segformer_b2_teacher_student_best.pt
python scripts/check_setup.py --verify-sha256 --load-models --device cpu
python -B -m unittest tests.test_tinea_segformer tests.test_wound_analysis tests.test_fungal_wound_overlap tests.test_photo_analysis_route tests.test_portable_policy
```

새 가중치를 다시 내보내면 `weights/manifest.json`의 크기와 SHA-256도 갱신해야 합니다. 이미 실행 중인 서버는 재시작 후 새 모델을 로드합니다.

## 전환 검증 (2026-10-18)

Windows / Python 3.10 / PyTorch 2.5.1 / Transformers 5.14.1에서 관련 회귀 테스트 87개를 통과했습니다. CPU에서 활성 네트워크 4개와 카메라 보정을 로드하고 manifest의 모든 SHA-256을 확인했습니다.

원래 학습 구현과 새 런타임의 테스트 75장 및 원본 크기 crop 3장, 총 78개 확률 맵을 비교했습니다. 서로 다른 PyTorch·Transformers 버전의 GPU FP32 계산에서 최대 확률 차이는 0.000322였고 임계값 경계 픽셀 30개가 달랐습니다. 새 런타임의 테스트 crop 평균 Dice/IoU는 0.603483/0.490283, 전체 픽셀 Dice/IoU는 0.694421/0.531888로 재현됐습니다. 모델 텐서는 내보낼 때 값을 변경하지 않았습니다.

실제 사진 3장으로 YOLO 발 검출부터 Student·상처 모델·파란색 무좀 표시까지 검사했습니다. 이 중 발 검출이 안 된 한 장은 기존 전체 사진 대체 경로를 사용했습니다. 서버 처리 전체에 대한 새 정답 성능 측정을 뜻하지는 않습니다.
