# Discord 커뮤니티 출범 계획

> **상태:** 초안 · **작성일:** 2026-10-06 · **플랫폼:** Discord (확정)
> **선행 PR:** [#14](https://github.com/trillion-labs/trida-stack/pull/14) (문서·자동화)

GitHub과 Hugging Face의 문의를 Discord에서 받아 대응하되, **답은 기준 시스템에 남기는**
운영 체계를 세우고 채널을 여는 작업입니다. 실행 순서, 완료 기준, 그리고 **열면 안 되는
조건**까지 적습니다.

---

## Context

trida-stack은 2026-09-28에 공개됐고 모델(`trillionlabs/Trida2.0-4B`)은 2026-09-19부터
공개돼 다운로드 106건이 쌓였습니다. 그런데 사용자가 질문할 곳이 사실상 없습니다 —
공개 시점에 README에는 `CONTRIBUTING.md`·`SECURITY.md` 링크조차 없었고, Discussions는
꺼져 있었으며, PR #4는 **138시간째 아무 응답이 없습니다.**

메인테이너 쪽 사정도 분명합니다. 3명이 연구와 병행하므로, 사람이 GitHub·HF·채팅 세 곳을
각각 들여다보는 운영은 지속되지 않습니다. **알림을 한 곳(Discord)으로 모으고 거기서
분류**하되, 답은 검색 가능한 곳에 남기는 구조가 필요합니다.

#14가 문서와 자동화를 준비했습니다. 이 문서는 **그 위에서 실제로 채널을 여는** 계획입니다.

## Current State (검증 완료 2026-10-06)

### 준비된 것 — PR #14

| 파일 | 역할 |
|---|---|
| `COMMUNITY.md` | 채널 맵, 응답 기대치. **채팅 행은 비어 있음** |
| `docs/community/CHANNEL_MATRIX.md` | 문의 유형별 기준 채널, 전환 규칙, 알림 설계 |
| `docs/community/CHANNEL_POLICY.md` | 최소 채널 구조, 역할·권한, 집행 |
| `docs/community/MODERATOR_RUNBOOK.md` | S1~S4 등급, 8단계 승격 절차, 모의훈련 4종 |
| `docs/community/LABELS.md` | 라벨의 뜻과 종료 기준 |
| `.github/workflows/community-check.yml` | 필수 파일·신고 경로 검증 (CI 통과 중) |
| `.github/workflows/stale-unanswered.yml` | 평일 09:00 KST, 48h 무응답 알림 |
| `tools/notify_stale.py` | 위 워크플로의 구현. 웹훅 URL이 설정된 쪽으로 전송 |
| `.github/ISSUE_TEMPLATE/community_to_issue.yml` | 채팅→이슈 전환 폼 |

적용 완료: GitHub Discussions 활성화, 라벨 `triage`·`community` 추가.

### 없는 것

| 항목 | 확인 방법 | 결과 |
|---|---|---|
| Discord 서버 | — | **없음** |
| 저장소 시크릿 `DISCORD_WEBHOOK_URL` | `gh secret list` | **비어 있음** |
| 저장소 웹훅 (GitHub→Discord) | `gh api .../hooks` | **0개** |
| HF 웹훅 | HF 저장소 설정 | **미설정** |
| `GOVERNANCE.md` | — | **없음** |
| 역할 담당자 배정 | `CHANNEL_POLICY.md` | **미지정** |
| 신고 메일함 수신 확인 | — | **미확인** ← 출범 보류 조건 |

`notify_stale.py`는 지금 돌아가지만 **웹훅이 없어 Actions 로그로만 출력**됩니다
(`tools/notify_stale.py:65-73`). 실패하지 않고 조용히 아무 데도 안 갑니다.

---

## 출범 보류 조건 (Launch-hold)

**하나라도 미충족이면 공개 초대를 열지 않습니다.** 문서가 준비된 것과 운영이 가능한 것은
다릅니다.

| # | 조건 | 확인 방법 | 현재 |
|---|---|---|---|
| H1 | 비공개 신고가 **실제로 수신**되는가 | `security@trillionlabs.co`·`conduct@trillionlabs.co`로 테스트 메일 발송 → 각각 **2명 이상** 수신 확인 | ❌ 미확인 |
| H2 | 관리자 계정 **복구**가 되는가 | 서버 소유자 외 관리자 2인, 전원 MFA, 복구 경로 문서화 | ❌ 미확인 |
| H3 | 휴가 때 **대신할 운영자**가 있는가 | 역할별 백업 1인 지정 | ❌ 미지정 |

H1이 가장 중요합니다. 행동강령에 신고 주소가 적혀 있는데 그 메일함이 죽어 있으면,
사고가 났을 때 받을 곳이 없습니다. `community-check.yml`은 **주소가 적혀 있는지만**
검사하며 메일 도착은 검사할 수 없습니다.

> H1~H3 확인 결과를 이 표에 날짜와 확인자로 기록합니다.

---

## Proposed Change

### 알림 경로 — 무엇이 어디로 가는가

```
GitHub 이벤트 ─┬─ release, repository_advisory ──[저장소 웹훅]──┐
               │                                                 │
               └─ help wanted 라벨 ───────────[Actions 필터]──────┼──> Discord
                                                                  │
48h 무응답 ────── stale-unanswered.yml (구현 완료) ───────────────┤
                                                                  │
HF Discussion ── HF 웹훅 ─────────────────────[형식 확인 필요]────┘

답장 방향: Discord에서 보고 → GitHub/HF로 이동해서 작성 (플랫폼 제약)
```

**기술적 제약 두 가지 — 나중에 발견하지 말고 지금 적어 둡니다:**

1. **GitHub 저장소 웹훅은 이벤트 타입으로만 구독하며 라벨 필터가 없습니다.**
   `help wanted`만 보내려면 웹훅이 아니라 Actions가 필요합니다 (T2.3).
2. **Discord에서 GitHub으로 답장하는 건 사실상 불가능합니다.** 알림은 Discord에서 보고,
   답은 GitHub으로 넘어가서 작성합니다. 우회 불가능하며, 승격 규율이 그래서 필요합니다.

### 왜 "승격"이 선택이 아니라 요건인가

`M-001 첫 사람 응답시간`은 **GitHub 댓글 타임스탬프**로 측정합니다
(`tools/collect_metrics.py`, PR #13). Discord에서만 답하면:

- 실제로 10분 만에 도왔어도 → 지표상 **미응답**
- `M-002 미응답률`이 실제보다 높게 나오고
- `M-003 SLA 달성률`은 측정 불가

지표가 틀리는 것보다 큰 문제는 **지식 소실**입니다. 가이드가 채팅 채널의 주요 위험으로
꼽는 항목이 정확히 그것입니다 — 지식 소실, 검색·보존 제한.

`stale-unanswered.yml`이 안전망입니다. Discord에서만 답한 항목은 누군가 GitHub에
기록할 때까지 **평일마다 계속 올라옵니다.**

---

## Implementation Details

### Phase 0 — 선행 (사람만 가능)

| # | 작업 | 담당 | 완료 기준 |
|---|---|---|---|
| **T0.1** | PR #14 머지 | usik-luke | `main`에 커뮤니티 문서·워크플로 반영 |
| **T0.2** | 역할 담당자 배정 | 팀 | 커뮤니티 리드 / 기술 당번 / 모더레이터 / 보안 담당 — 4역할을 3명에 배분, 각 역할 백업 1인 |
| **T0.3** | **H1** 신고 메일함 테스트 | 보안 담당 | 두 주소에 테스트 발송, 각 2명 이상 수신 확인, 결과를 위 표에 기록 |

T0.3이 실패하면 **T2 이후 전체가 중단**됩니다.

### Phase 1 — Discord 서버 구축

| # | 작업 | 완료 기준 |
|---|---|---|
| **T1.1** | 서버 생성, 소유권 설정 | 소유자 1 + 관리자 2, 전원 MFA (**H2**) |
| **T1.2** | 최소 채널 생성 | `CHANNEL_POLICY.md`의 7개 채널. 그 이상 만들지 않음 |
| **T1.3** | `#rules` 고정 | `CHANNEL_POLICY.md`의 허용/금지/집행 전문 + 신고 주소 |
| **T1.4** | 규칙 동의 게이트 | 동의 전 쓰기 불가. 접근성 영향 확인 |
| **T1.5** | 초대 링크 정책 | 만료·사용범위 결정. 공개 위치는 README와 `COMMUNITY.md`로 **제한** |
| **T1.6** | 모의훈련 4종 | `MODERATOR_RUNBOOK.md` 체크리스트 전부 수행 |

채널 구조 (`CHANNEL_POLICY.md`에서, 그대로):

```
START-HERE   #rules  #announcements
PROJECT      #general  #help  #contributors
PRIVATE      #moderators
```

`#help` 질문은 **thread로 분리**합니다. `#general`에서 기술 결정을 확정하지 않습니다.

### Phase 2 — 알림 배선

| # | 작업 | 구현 | 완료 기준 |
|---|---|---|---|
| **T2.1** | 48h 알림 연결 | Discord `#moderators` 웹훅 생성 → 저장소 시크릿 `DISCORD_WEBHOOK_URL` | `gh workflow run stale-unanswered.yml` 실행 시 Discord에 **도착 확인** |
| **T2.2** | 릴리스·보안 공지 | 저장소 Settings → Webhooks → `<웹훅URL>/github`, 이벤트 **`release`, `repository_advisory`만** | 테스트 릴리스로 도착 확인 |
| **T2.3** | `help wanted` 알림 | 신규 `.github/workflows/notify-labeled.yml` — `issues: [labeled]` 트리거, 라벨 필터 후 같은 웹훅으로 전송 | 라벨 부착 시 1건 도착, 다른 라벨은 무음 |
| **T2.4** | HF Discussion 알림 | HF 저장소 Settings → Webhooks | **선행 조사 필요** — HF는 범용 웹훅이라 Discord 형식과 바로 안 맞을 수 있음. 중계가 필요하면 별도 과제로 분리 |

**T2.2에서 이벤트를 전부 켜지 않습니다.** 모든 commit·PR·댓글을 보내면 사람이 읽지
않습니다. 5종만 보냅니다 — 릴리스, 보안 공지, 48h 무응답, `help wanted`, 모델 카드 변경.

**양방향 복제 봇은 쓰지 않습니다.** 삭제·편집·차단 상태와 개인정보가 어긋납니다.

### Phase 3 — 콘텐츠 준비 (빈 커뮤니티를 열지 않음)

| # | 작업 | 완료 기준 |
|---|---|---|
| **T3.1** | FAQ 10건 | Discussions Q&A에 질문+답변 게시, **답변 표시(marked)** |
| **T3.2** | `good first issue` 3~5건 | 각 이슈 본문에 배경·파일 위치·예상 변경 크기·검증 방법·질문할 곳 — 다섯 가지가 없으면 라벨 붙이지 않음 |
| **T3.3** | PR #4 응답 | 138시간 무응답 해소. 분류만 해도 됨 |
| **T3.4** | `COMMUNITY.md` 채팅 행 채우기 | 초대 URL 삽입, README 동기화 |

### Phase 4 — 제한 공개 → 공개

| # | 작업 | 완료 기준 |
|---|---|---|
| **T4.1** | 제한 공개 | 기존 사용자·협력기관 소수에게 먼저. 신규 참여자가 **안내만 보고** 올바른 채널에 질문하는지 관찰 |
| **T4.2** | 전 경로 시험 | 초대 링크 / 행동강령 신고 / Issue Form / Discussion 답변 표시 / 채팅→이슈 전환을 **끝까지** |
| **T4.3** | 기준선 기록 | 회원 수가 아니라 — 질문 수, 첫 사람 응답시간, 답변 완료율, GitHub 전환 수, 운영자 시간 |
| **T4.4** | 공개 | README·`COMMUNITY.md`·모델 카드에 초대 링크 |

---

## Acceptance Criteria

1. `gh secret list`에 `DISCORD_WEBHOOK_URL`이 존재한다
2. `gh workflow run stale-unanswered.yml` 수동 실행 시 Discord에 메시지가 **도착한다** (Actions 로그 출력이 아니라)
3. 테스트 릴리스 발행 시 `#announcements`에 1건 도착한다
4. `help wanted` 라벨 부착 시 1건 도착하고, 다른 라벨 부착 시 **아무것도 오지 않는다**
5. 저장소 웹훅의 구독 이벤트가 `release`, `repository_advisory` **2종뿐이다** (`gh api .../hooks`로 확인)
6. Discussions Q&A에 답변 표시된 FAQ가 10건 이상이다
7. `good first issue` 라벨이 붙은 이슈가 3건 이상이고, **각각 다섯 요소를 모두 포함한다**
8. 로그아웃한 브라우저에서 README → Discord 초대까지 **1클릭**으로 도달한다
9. `community-check.yml`이 통과한다
10. H1·H2·H3가 모두 충족되고 **위 표에 확인 일자와 확인자가 기록되어 있다**
11. PR #4의 무응답이 해소되었다
12. `#help` 질문 1건이 thread → GitHub Issue/Discussion으로 **승격된 사례가 있다** — 아무도 해보지 않은 규칙은 아직 관행이 아닙니다

---

## Testing Plan

| 계층 | 무엇을 | 건수 |
|---|---|---|
| 단위 | `notify_stale.py` 목적지 선택 (Discord 설정 / Slack 설정 / 둘 다 없음) | +3 |
| 통합 | `workflow_dispatch`로 48h 알림 실제 발송 → Discord 도착 | +1 |
| 통합 | 테스트 릴리스 → `#announcements` 도착 | +1 |
| 통합 | `help wanted` 라벨 → 1건 / 다른 라벨 → 0건 | +2 |
| E2E | 로그아웃 상태에서 README → 초대 → 규칙 동의 → `#help` 질문 → 승격 | +1 |
| 수동 | 모의훈련 4종 (`MODERATOR_RUNBOOK.md`) | +4 |

`notify_stale.py`의 목적지 선택 로직은 **현재 테스트가 없습니다.** T2.1 전에
`tools/test_notify_stale.py`를 추가합니다 — 지금은 웹훅이 없어 "출력만" 경로밖에
실행되지 않으므로, 전송 경로는 테스트로만 검증할 수 있습니다.

---

## Rollback Plan

| 상황 | 되돌리기 |
|---|---|
| 알림이 과도함 | 저장소 Settings에서 웹훅 비활성화. 코드 변경 불필요 |
| 48h 알림이 시끄러움 | `DISCORD_WEBHOOK_URL` 시크릿 삭제 → 로그 출력으로 되돌아감, 워크플로는 실패하지 않음 |
| 서버 운영 불가 | 초대 링크 폐기 → `#announcements`에 중단 공지 → 30일 읽기 전용 → 보관 |
| 사건 발생 | `MODERATOR_RUNBOOK.md` S1~S4 |

채널을 닫아도 **결정과 FAQ가 GitHub에 남아 있으면 이전 비용이 작습니다.** 이것이 승격
규율의 두 번째 이유입니다.

---

## Effort Estimate

| Phase | 작업 | 사람 시간 |
|---|---|---|
| 0 | 머지, 역할 배정, 메일함 테스트 | ~2h (+ 메일 응답 대기) |
| 1 | 서버 구축, 규칙, 모의훈련 | ~4h |
| 2 | 웹훅 배선, 도착 확인 | ~2h (T2.4 HF 조사 별도) |
| 3 | FAQ 10건, good first issue 3~5건 | ~6h ← **가장 큼** |
| 4 | 제한 공개, 관찰, 공개 | ~3h + 관찰 1~2주 |
| | **합계** | **~17h + 관찰 기간** |

Phase 3이 가장 큽니다. FAQ와 첫 기여 이슈는 **실제로 처리 가능한 것**이어야 하므로
기계적으로 채울 수 없습니다.

T2.3(`notify-labeled.yml`)과 `test_notify_stale.py`는 에이전트 작업 가능 — 합계 ~1h.

---

## Files Reference

| 파일 | 변경 |
|---|---|
| `docs/community/LAUNCH_PLAN.md` | 이 문서. H1~H3 확인 결과를 여기에 기록 |
| `COMMUNITY.md` | T3.4 — 채팅 행에 초대 URL |
| `README.md` | T3.4 — Discord 링크 추가 |
| `docs/community/CHANNEL_POLICY.md` | T0.2 — 역할 담당자 핸들 |
| `docs/community/CHANNEL_MATRIX.md` | T1.1 — 채널 운영 기록 표 (소유 계정, 관리자 2인) |
| `.github/workflows/notify-labeled.yml` | **신규** — T2.3 |
| `tools/test_notify_stale.py` | **신규** — 목적지 선택 테스트 |
| 저장소 Settings → Secrets | **신규** — `DISCORD_WEBHOOK_URL` |
| 저장소 Settings → Webhooks | **신규** — `release`, `repository_advisory` |

---

## Out of Scope

- **Slack** — Discord로 확정. `notify_stale.py`의 Slack 분기는 코드에 남지만 사용하지 않습니다
- **메일링리스트, 별도 포럼** — 지금 규모에 불필요
- **채팅 → GitHub 양방향 쓰기** — Discord 플랫폼 제약. 승격은 사람이 합니다
- **음성 채널·오피스아워·밋업** — 운영이 안정된 뒤
- **`GOVERNANCE.md`** — 별도 과제. 채널 출범과 독립
- **커뮤니티 지표 수집** — 사용자가 직접 관리
- **`collect_metrics.py`의 HF Discussion 확장** — 실제 문의가 HF로 들어오기 시작하면 별도 과제

---

## Related

- PR [#14](https://github.com/trillion-labs/trida-stack/pull/14) — 커뮤니티 문서·자동화 (선행)
- PR [#13](https://github.com/trillion-labs/trida-stack/pull/13) — 메트릭 (M-001~M-003이 승격 규율에 의존)
- PR [#12](https://github.com/trillion-labs/trida-stack/pull/12) — 가중치 Apache-2.0 (모델 카드 대기)
- PR [#4](https://github.com/trillion-labs/trida-stack/pull/4) — 138시간 무응답, T3.3 대상
- `docs/metrics/README.md` — 지표 정의와 관찰 기간
