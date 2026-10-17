# Feetfit AI API

Feetfit의 통합 FastAPI 서버입니다. 발 사진·족저 압력·온습도 분석 기능은
`AI-Server-Eunseo` 구현을 기준으로 하고, 신발 추천·리뷰 근거·요약 생성 기능은
`Feetfit_AI`(Minji) 구현을 기준으로 합니다. Python 패키지 호환 범위와 실행 방식도
Minji 프로젝트를 기준으로 유지합니다.

`app/services/shoe/`와 신발 API 계약은 Minji 영역입니다. 그 밖의 사진, 무좀,
상처, 무지외반, 발바닥 접촉, 압력 히트맵 및 환경 분석은 Eunseo 영역입니다.

## Structure

```text
app/
  main.py                       FastAPI entry point
  api/
    router.py                   Root API router
    routes/
      reports.py                Legacy foot reports + Minji shoe-recommendation endpoint
      shoes.py                  Shoe detail summary generation trigger endpoint
      photo_analysis.py         Integrated photo-analysis intake
      pressed_regions.py        Plantar footprint/contact analysis
      pressure_heatmaps.py      Pressure heatmap and balance analysis
      environment.py            Temperature/humidity analysis
  core/
    config.py                   Environment and app settings
    security.py                 Swagger Bearer token security scheme
    weights.py                  Centralized model weight paths
  prompts/
    shoe_fit_comment_prompts.py SYSTEM_PROMPT / USER_PROMPT_TEMPLATE for Ollama
  schemas/
    reports.py                  Tinea pedis / hallux valgus request schemas
    shoes.py                    Shoe recommendation batch request/forward schemas
    shoe_server.py              Minji Feetfit_Server internal read contracts
    shoe_fit_comment.py         Minji Ollama summary response contract
  services/
    tinea_analysis.py           Eunseo foot/tinea segmentation and scoring
    integrated_foot_analysis.py Eunseo ArUco geometry and integrated analysis
    pressure_heatmap_analysis.py Eunseo pressure analysis
    wound_analysis.py           Eunseo wound segmentation
    shoe/
      shoe_server_client.py       Minji Feetfit_Server context/callback client
      shoe_embedding.py           Minji BGE-M3 embedding/cache/cosine ranking
      shoe_feature_rules.py       Minji semantic review-search needs
      shoe_fit_policy.py          Minji RunRepeat fit policy
      shoe_recommendation.py      Minji fitScore/risk/review recommendation
      shoe_fit_comment_service.py Minji point/review summary generation
weights/                        Model weights, ignored by git
```

This is a common FastAPI layout: route definitions stay in `api`, shared
configuration and security live in `core`, request/response contracts live in
`schemas`, LLM prompt text lives in `prompts`, and business helpers live in
`services`.

## Run

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Using the virtual environment's Python directly works in both PowerShell and
`cmd.exe`; shell activation is optional.

Swagger:

```text
http://localhost:8000/docs
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Local model assets

Model weights, generated images, embedding caches, local audit artifacts, and
`.env` are intentionally excluded from Git. Copy the required weights into the
local `weights/` directory or configure their paths in `.env`; do not commit
them to GitHub. Start from `.env.example`, whose values are placeholders.

### Tests

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest -q
```

### ngrok development tunnel

Install the ngrok CLI and configure its token outside this repository once:

```powershell
ngrok config add-authtoken <YOUR_NGROK_TOKEN>
```

Then start FastAPI and ngrok together:

```powershell
.\venv\Scripts\python.exe scripts\run_ngrok.py
```

The script reuses an existing service on port 8000 or starts one with the same
virtual-environment interpreter. Private keys, ngrok credentials, and `.env`
must remain local.

Click `Authorize` in Swagger and paste the JWT Bearer token before calling any
endpoint below (all of them forward the same token to Feetfit_Server).

The legacy tinea endpoint is:

```text
POST /api/reports/tina-pedis
```

Input fields:

```text
measurementSessionId  default: 1
leftFootImage         one original left foot image
rightFootImage        one original right foot image
```

The server analyzes both images, creates:

```text
request                 generated scores and temporary descriptions
suspiciousAreaMapImage  left + right circular suspicion maps combined side by side as PNG
originalFootImage       left + right cutout photo overlays combined side by side as PNG
```

Each side is cropped to the detected foot region with transparent PNG background,
then the left and right images are combined side by side for the tinea backend.

The safety scores use the more conservative value between the two feet.
When `TINEA_SLIDING_WINDOW_ENABLED=true`, the tinea model also runs on overlapping
tiles inside the detected foot bbox and merges tile probabilities with the
full-image probability map to improve small-region detection.
The tinea contrast/red enhancement options apply only to the tinea model input;
hallux valgus landmark inference keeps the original image preprocessing.

Then it forwards the multipart request to:

```text
http://127.0.0.1:8080/api/reports/tina-pedis
```

The tinea description fields and hallux `scoreAnalysisText` are generated with
the OpenAI Responses API when `OPENAI_API_KEY` is configured. If the key is not
set or the API call fails, the server falls back to local rule-based Korean
summary text and still forwards the report request.

현재 분석 모델 manifest(`weights/manifest.json`)에 기록된 파일은 다음과 같습니다.

```text
<project>/weights/foot_seg_yolo11n_best.pt
<project>/weights/tinea_segformer_b2_teacher_student_best.pt
<project>/weights/wound_segformer_experiment_c_best.pt
<project>/weights/plantar_contact_segformer_b0_real_finetune_brightness2_best.pt
<project>/weights/camera_calibration.npz
```

추가 로컬 모델을 사용하는 기능은 `.env.example`의 경로 설정을 확인하세요.
가중치 바이너리와 실제 `.env`는 Git에 올리지 않습니다.

The shoe-recommendation endpoints additionally require:

- Feetfit_Server's internal shoe-analysis endpoints. The incoming user Bearer token
  is forwarded unchanged together with `X-Internal-Api-Key`. Configure
  `FEETFIT_SERVER_INTERNAL_API_KEY` to the same service key as Feetfit_Server;
  missing or blank keys fail before any request is sent. The shared DB is not used
  as a fallback when an internal API request fails.
- [Ollama](https://ollama.com) running locally with a pulled model:

  ```powershell
  ollama pull qwen2.5:7b-instruct
  ```

  (model name is configurable via `OLLAMA_MODEL`; defaults to `qwen2.5:7b-instruct`)

## Endpoints

### `POST /api/reports/tina-pedis`

Upload one left + one right foot image. The server analyzes both, creates:

```text
request                 generated scores and temporary descriptions
suspiciousAreaMapImage  left + right circular suspicion maps combined side by side
originalFootImage       left + right red/blue photo overlays combined side by side
```

The suspicion map is generated only from the segmentation mask used in the photo
overlay. The safety scores use the more conservative value between the two feet.
Then it forwards the multipart request to `TINEA_REPORT_ENDPOINT`.

The description fields are generated through the OpenAI Responses API when it
is configured. If generation is unavailable, local Korean rule-based fallback
text is used so report forwarding can still complete.

```bash
curl -X POST "http://localhost:8000/api/reports/tina-pedis" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "leftFootImage=@left-foot.png;type=image/png" \
  -F "rightFootImage=@right-foot.png;type=image/png"
```

## Single-photo integrated analysis

`POST /api/reports/integrated-foot-analysis` accepts one calibrated two-foot
photo and runs the complete pipeline:

```text
one 1280x720 photo
  -> lens undistortion
  -> ArUco board orientation and planar calibration
  -> one YOLO foot-boundary inference
  -> mask rectification and left/right split
  -> foot length and straight-line MTP-zone ball width
  -> tinea analysis with the same per-foot masks
  -> hallux-valgus analysis with the same per-foot masks
  -> existing tinea and hallux report backends
```

ArUco detects the board geometry before converting the YOLO masks to the
metric plane. The YOLO network itself runs only once on the lens-corrected
source frame. Both disease analyzers consume those shared masks and do not run
their own foot-boundary YOLO inference on this endpoint.

The validated ArUco implementation and camera calibration are loaded from the
separately cloned repository configured by `ARUCO_SOURCE_DIR` (default:
`../ArUco-marker-code`, configurable through `ARUCO_SOURCE_DIR`). Keep that clone at the matching validated revision and
make sure its `models/hybrid_best_1280x720.npz` file exists.

```bash
curl -X POST "http://localhost:8000/api/reports/integrated-foot-analysis" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "footImage=@two-feet-with-aruco.jpg;type=image/jpeg"
```

The response contains `analysis.feet.left/right` with `lengthMm`,
`ballWidthMm`, tinea safety scores, and hallux angle, plus the response from
each existing report backend. HTTP `200` means both reports were accepted;
`207` means analysis completed but at least one backend returned an error.

Important capture constraints:

- The default lens model is valid only for an unmodified 1280x720 full frame.
- Use `DICT_4X4_50` marker IDs 0 through 5 and the physical dimensions in
  the `.env` reference below.
- `image_left` and `image_right` in the ArUco code are board bays, not anatomy.
  Confirm the fixed capture protocol and set
  `ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE` to `left` or `right` accordingly. The
  bundled example protocol maps the canonical image-left bay to the left foot.
- `ballWidthMm` is a straight line across the silhouette's MTP proxy zone, not
  foot girth.
- Always check `measurementValid` and `measurementInvalidReasons`; disease
  analysis and metric-measurement quality are reported separately.

### `POST /api/reports/hallux-valgus`

Upload one left + one right foot image. The server extracts the foot outline,
computes the hallux valgus angle (HVA) per foot, and renders an analysis image
with 3 keypoints and connecting lines drawn on the outline. Forwards to
`HALLUX_VALGUS_REPORT_ENDPOINT` the same way as the tinea pedis endpoint.

```bash
curl -X POST "http://localhost:8000/api/reports/hallux-valgus" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "measurementSessionId=1" \
  -F "leftFootImage=@left-foot.jpeg;type=image/jpeg" \
  -F "rightFootImage=@right-foot.jpeg;type=image/jpeg"
```

### `POST /api/reports/foot-type-text`

Feetfit_Server의 measurement session이 `COMPLETED`로 commit된 뒤 Server가
해당 세션의 DB 분석 결과를 다시 읽어 자동으로 호출하는 내부 전용
endpoint입니다. 클라이언트가 `typeText`나 `careTips`를 작성하여 보내지
않습니다. Server는 exact-session facts와 해당 facts의 SHA-256
`factsHash`를 함께 전달합니다.

```json
{
  "measurementSessionId": 30,
  "measurementStatus": "COMPLETED",
  "factsHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "analysis": {
    "archType": "LOW",
    "footWidthType": "WIDE",
    "pressureBalanceType": "BALANCED",
    "measuredLeftFootSizeMm": 253.0,
    "measuredRightFootSizeMm": 251.0,
    "leftFootWidthMm": 101.0,
    "rightFootWidthMm": 100.0,
    "leftPressurePercent": 49.0,
    "rightPressurePercent": 51.0,
    "plantarFootprintAnalysisText": "발바닥 중앙부와 뒤꿈치의 압력 분포 차이가 관찰됩니다."
  }
}
```

GPT는 `ARCH`, `WIDTH`, `PRESSURE_BALANCE`의 명시적 분류 중 신발 선택에
가장 유용한 evidence ID만 고릅니다. 최종 한국어는 해당 evidence에 고정된
검증 문구로 렌더링되므로 GPT가 평발이나 발볼 특성을 새로 추측할 수
없습니다. OpenAI 호출이 실패하거나 잘못된 ID를 반환하면 `ARCH → WIDTH →
PRESSURE_BALANCE` 순서의 로컬 fallback을 사용합니다. 원시 발 길이와 너비는
추적용일 뿐, 그 숫자만으로 아치나 발볼 유형을 분류하지 않습니다.
OpenAI에는 원시 측정값이 아닌 검증된 candidate evidence만 전달하며,
최종 문구는 `"이번 측정에서는"`으로 시작하지 않도록 응답 계약에서도
검증합니다.

AI는 DB를 수정하지 않고 다음 pure-generation 응답만 반환합니다.

```json
{
  "measurementSessionId": 30,
  "factsHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "typeText": "발의 아치가 낮아 발바닥이 넓게 닿는 편이에요. 오래 걷거나 서 있으면 피로가 커질 수 있어 아치를 잘 받쳐주는 신발이 더 편안할 수 있어요.",
  "evidenceId": "ARCH_LOW",
  "source": "OPENAI"
}
```

Server는 응답의 `measurementSessionId`와 `factsHash`를 현재 DB facts와 다시
검증하고, 동일 완료 세션의 `typeText`가 아직 없을 때만 `typeText`를
저장합니다. `careTips`는 읽거나 덮어쓰지 않습니다. 이후 신발 목록은
Server의 `GET /api/reports/foot-type-text`로 저장된 문구를 조회합니다.

```bash
# 정상 사용 시 Server가 자동 호출하며, 아래는 로컬 내부-계약 점검용 예시입니다.
curl -X POST "http://localhost:8000/api/reports/foot-type-text" \
  -H "Authorization: Bearer <SERVER_MINTED_TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"measurementSessionId":30,"measurementStatus":"COMPLETED","factsHash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","analysis":{"archType":"LOW"}}'
```

### `POST /api/reports/shoe-recommendations`

```json
{ "measurementSessionId": 30 }
```

Uses the forwarded Bearer token to page through Feetfit_Server's internal
recommendation-context endpoint. The response is fixed to the requested completed
measurement session and contains its foot analyses, MUSINSA-only reviews, and raw
RunRepeat lab measurements. The Phase-D scorer computes **fitScore + riskLevel
(FOREFOOT/HEEL/INSOLE)** from the relationship between the one requested session's
foot measurements and each shoe's real RunRepeat metrics. Review sentiment never
changes score or risk. Missing characteristics are not synthesized; present
components are reweighted, and a body area with zero real components fails closed.
Every catalog shoe is returned, including shoes with zero reviews (`reviewIds: []`).
BGE-M3 selects at most three shoe-local MUSINSA candidates per body part.

The numerical policy and semantic threshold are explicitly
`TEMPORARY_HEURISTIC` / `NOT_CLINICALLY_VALIDATED`; weights, allowances and risk/
pressure/humidity thresholds are environment-backed settings rather than clinical
claims. The shoe comparison feature is intentionally out of scope.

**No LLM call happens here.** The batch deterministically renders an
evidence-backed `pointSummary` from the shoe's independent MUSINSA review
consensus. When reviews cannot support product copy, it uses only the calculated
user/product compatibility as a non-null fallback and never invents styling or
sizing facts. It also creates all three per-area `reviewSummary` values from the
same session's pressure, dimensions, foot environment, compatibility components,
and repeated review evidence. The complete result is forwarded to
`FEETFIT_SERVER_RECOMMENDATION_ENDPOINT`, so a normal detail request is a single
database read and does not wait for Ollama.

```bash
curl -X POST "http://localhost:8000/api/reports/shoe-recommendations" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"measurementSessionId": 30}'
```

### `POST /api/shoes/summaries/generate`

```json
{ "shoeId": 220, "measurementSessionId": 30 }
```

이전 배치에서 요약이 비어 있는 데이터만 보완하기 위한 호환 API입니다. 신규
추천 배치는 이미 완성된 요약 네 개를 함께 저장하므로 정상 흐름의 상세 조회에서는
이 API를 호출하지 않습니다. 호출되는 경우 저장된 동일 측정 세션의 추천 근거와
신발 특성을 읽고 Server의 기존 요약 저장 DTO와 같은 형태를 반환합니다.

```json
{
  "measurementSessionId": 30,
  "pointSummary": "이 신발의 주요 착화 특성을 자연스럽게 정리한 문장입니다.",
  "reasons": [
    {"reasonType": "FOREFOOT", "reviewSummary": "발볼 요약", "reviewIds": [1]},
    {"reasonType": "HEEL", "reviewSummary": "뒤꿈치 요약", "reviewIds": [2]},
    {"reasonType": "INSOLE", "reviewSummary": "깔창 요약", "reviewIds": [3]}
  ]
}
```

이 경로는 Feetfit_Server 저장 callback을 호출하지 않습니다. Server가 응답의
`measurementSessionId`와 세 reason 계약을 검증해 직접 저장한 뒤 같은 상세 응답에
반영합니다. 같은 세션/신발의 동시 요청은 하나의 Ollama 결과를 공유합니다.
Ollama 전송 실패·혼잡 또는 `OLLAMA_DIRECT_SUMMARY_DEADLINE_SECONDS` 초과 시에는
동일 근거로 만든 non-null 결정론적 문장을 HTTP `200`으로 반환합니다.

```bash
curl -X POST "http://localhost:8000/api/shoes/summaries/generate" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"shoeId":220,"measurementSessionId":30}'
```

### `POST /api/shoes/summaries` (legacy async trigger)

```json
{ "shoeId": 220, "measurementSessionId": 30, "waitForImmediate": true }
```

Must be called by Feetfit_Server when a shoe detail view finds an absent
`pointSummary` or per-area `reviewSummary`; this AI endpoint cannot initiate that
hook itself. `waitForImmediate` defaults to `false`, preserving the original
fire-and-forget `202` contract. When it is `true`, the request waits at most four
seconds for steps 1-2 to complete in Feetfit_Server and returns
`202 {"accepted":true,"deduplicated":false,"immediateSaved":true}`. Ollama
refinement still runs after the response as a background task:

1. Read the **already-computed** fitScore/riskLevel/title and evidence review
   texts from Feetfit_Server's JWT-scoped summary-context endpoint. Nothing is
   recalculated (no embeddings and no foot-state lookup). The exact completed
   `measurementSessionId` is required; no current/older-session fallback is used.
2. Persist a deterministic typed-evidence summary immediately. This gives old
   pending rows and per-area summaries a safe provisional value before GPU
   generation completes; it never changes fitScore/riskLevel.
3. Call Ollama (`OLLAMA_MODEL`, temperature `OLLAMA_TEMPERATURE`) with the exact
   Pydantic JSON Schema to semantically remove unrelated BGE candidates (the
   returned IDs must be a unique subset of the exact shoe/reason candidates,
   maximum three), and turn the facts into Korean `pointSummary` +
   `forefootSummary`/`heelSummary`/`insoleSummary`. A schema-invalid response gets
   one correction attempt using the original grounded prompt; a second contract
   failure uses the deterministic typed-evidence fallback. The prompt explicitly
   forbids inventing review content for body parts with zero evidence reviews,
   forbids medical-diagnosis language, and must not contradict the given
   riskLevel. Response is parsed as JSON with a regex-extraction fallback if the
   model wraps it in prose/markdown.
4. `POST` the refined summary to `{shoe_id}/summaries` on Feetfit_Server
   (`FEETFIT_SERVER_SUMMARY_SAVE_ENDPOINT_TEMPLATE`), forwarding the same Bearer
   token, exact measurementSessionId, summaries, and the validated final reviewIds.

Before generation, the AI also reads `GET /api/shoes/{shoe_id}/characteristics`
through `FEETFIT_SERVER_CHARACTERISTICS_ENDPOINT_TEMPLATE`. The returned RunRepeat
levels remain separate from subjective review evidence (for example cushioning,
shock absorption, and sole thickness are never treated as interchangeable).

Missing Server context, unrecoverable Ollama transport errors, and save callback
failures are logged and silently dropped. Schema-only failures use the grounded
fallback and are still saved. Once the Server detail-view hook is installed, the
next detail view can retry any dropped task because `pointSummary` stays `null`.

```bash
curl -X POST "http://localhost:8000/api/shoes/summaries" \
  -H "Authorization: Bearer <TOKEN>" \
  -H "X-Internal-Api-Key: <FEETFIT_SERVER_INTERNAL_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"shoeId": 220, "measurementSessionId": 30, "waitForImmediate": true}'
```

## `.env` reference

```text
TINEA_REPORT_ENDPOINT=http://127.0.0.1:8080/api/reports/tina-pedis
HALLUX_VALGUS_REPORT_ENDPOINT=http://127.0.0.1:8080/api/reports/hallux-valgus
FEETFIT_SERVER_RECOMMENDATION_ENDPOINT=http://127.0.0.1:8080/api/shoes/recommendations
FEETFIT_SERVER_SUMMARY_SAVE_ENDPOINT_TEMPLATE=http://127.0.0.1:8080/api/shoes/{shoe_id}/summaries
FEETFIT_SERVER_RECOMMENDATION_CONTEXT_ENDPOINT=http://127.0.0.1:8080/api/internal/shoe-analysis/recommendation-context
FEETFIT_SERVER_SUMMARY_CONTEXT_ENDPOINT_TEMPLATE=http://127.0.0.1:8080/api/internal/shoe-analysis/shoes/{shoe_id}/recommendation-summary-context
FEETFIT_SERVER_CHARACTERISTICS_ENDPOINT_TEMPLATE=http://127.0.0.1:8080/api/shoes/{shoe_id}/characteristics
SHOE_RECOMMENDATION_CONTEXT_PAGE_SIZE=100  # 1..200 (Feetfit_Server contract)
FEETFIT_SERVER_INTERNAL_API_KEY=<same-service-key-configured-on-Feetfit_Server>  # preferred
INTERNAL_API_KEY=<same-value>  # accepted shared-name fallback
REPORT_PROXY_TIMEOUT_SECONDS=60
FEETFIT_SERVER_CALLBACK_TIMEOUT_SECONDS=900  # callback floor; must be >= 900

# OpenAI report text generation
OPENAI_API_KEY=paste-your-openai-api-key-here
OPENAI_REPORT_MODEL=gpt-4.1-mini
OPENAI_REPORT_TIMEOUT_SECONDS=20
OPENAI_REPORT_TEXT_ENABLED=true
OPENAI_FOOT_TYPE_TEXT_ENABLED=true
FOOT_TYPE_PRESSURE_BALANCE_TOLERANCE_PERCENT=5
OPENAI_REPORT_INCLUDE_IMAGES=true

# Tinea analysis and visualization
COMBINED_IMAGE_MAX_HEIGHT=1600
COMBINED_IMAGE_GAP_PIXELS=16
FUNGAL_THRESHOLD=0.78
INFLAMMATION_THRESHOLD=0.77
TINEA_SLIDING_WINDOW_ENABLED=true
TINEA_SLIDING_WINDOW_TILE_SIZE=768
TINEA_SLIDING_WINDOW_OVERLAP=0.3
TINEA_SLIDING_WINDOW_PADDING=32
TINEA_SLIDING_WINDOW_MAX_TILES=12
TINEA_PREPROCESS_ENHANCE_ENABLED=true
TINEA_PREPROCESS_CONTRAST_GAIN=1.05
TINEA_PREPROCESS_CLAHE_CLIP_LIMIT=1.6
TINEA_PREPROCESS_RED_SATURATION_GAIN=1.10
TINEA_PREPROCESS_RED_VALUE_GAIN=1.02
MAX_DOT_AREA_FOR_SUSPICION_MAP=450
PHOTO_CUTOUT_BACKGROUND=true
PHOTO_CUTOUT_PADDING=28

# Single-photo ArUco pipeline
ARUCO_SOURCE_DIR=C:/path/to/aruco-marker-code
ARUCO_DICTIONARY=DICT_4X4_50
ARUCO_EXPECTED_IMAGE_WIDTH=1280
ARUCO_EXPECTED_IMAGE_HEIGHT=720
ARUCO_MARKER_SIZE_MM=20
ARUCO_MARKER_ROW_SPACING_MM=171
ARUCO_MARKER_COLUMN_SPACING_MM=140
ARUCO_FIXED_OFFSET_MM=113
ARUCO_IMAGE_LEFT_ANATOMICAL_SIDE=left

# BGE-M3 sentence embedding (shoe-recommendations batch)
SHOE_EMBEDDING_DEVICE=auto        # auto | cpu | cuda
SHOE_EMBEDDING_LOCAL_FILES_ONLY=true  # use the provisioned cache; disable only for the initial download
SHOE_MAX_CANDIDATE_REVIEWS_PER_REASON=40
SHOE_REVIEWS_PER_REASON=3                  # 1..3 (save payload contract)
SHOE_REVIEW_SEMANTIC_MIN_SCORE=0.42        # TEMPORARY_HEURISTIC
SHOE_RELEASE_EMBEDDING_MODEL_AFTER_BATCH=true
SHOE_RISK_LOW_MIN_SCORE=75
SHOE_RISK_MEDIUM_MIN_SCORE=50

# Ollama (shoe-summaries on-demand generation)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.0
OLLAMA_NUM_GPU=-1                 # GPU preferred; 0 forces CPU
OLLAMA_CPU_FALLBACK_ENABLED=true
OLLAMA_DIRECT_SUMMARY_DEADLINE_SECONDS=8.5
OLLAMA_KEEP_ALIVE=30m
OLLAMA_PREWARM_ON_STARTUP=false
```

`OLLAMA_KEEP_ALIVE`는 한 번 로드된 모델을 상세 탐색 중 상주시켜 warm 요청을 보통
수 초 안에 처리하도록 합니다. 재시작 직후부터 자연어 결과가 필요하면
`OLLAMA_PREWARM_ON_STARTUP=true`로 설정할 수 있지만, startup이 모델 로드를 마칠
때까지 길어질 수 있습니다. 특히 8 GB GPU에서는 Qwen 7B와 BGE-M3 추천 배치를
동시에 상주시킬 경우 VRAM 부족 가능성이 있으므로 추천 배치 실행 시점을 분리하거나
prewarm을 끄세요. Ollama 장애 시에도 direct API는 검증된 non-null fallback을
반환하지만 그 응답이 Ollama 문체라고 보장되지는 않습니다.

`SHOE_EMBEDDING_LOCAL_FILES_ONLY=true`는 이미 내려받은 BGE-M3 캐시를 사용하며
요청 중 Hugging Face 네트워크 확인을 하지 않습니다. 모델을 처음 내려받는 환경에서만
외부 통신이 가능한 상태로 `false`를 사용한 뒤, 운영 요청에서는 다시 `true`로 두세요.

All `FEETFIT_SERVER_*_ENDPOINT*` values must be set to the intended deployment
environment. Defaults are loopback-only and never point at a public Server. The
internal API key is mandatory: a blank `FEETFIT_SERVER_INTERNAL_API_KEY` fails
before any Server request and is never logged.
