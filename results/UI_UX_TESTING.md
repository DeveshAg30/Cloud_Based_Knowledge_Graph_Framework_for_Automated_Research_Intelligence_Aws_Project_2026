# UI/UX testing report

Testing of the deployed dashboard (https://main.d1gnkprv2sjfmv.amplifyapp.com/) on 15 October 2026. Browser: Microsoft Edge
(Chromium), driven by Playwright from a Windows laptop. Signed-in checks used a temporary Cognito user that was deleted afterwards.
Functional API coverage is in [EVALUATION.md](EVALUATION.md) section 9 (`scripts/smoke_test.py`, 13/13 checks).

## 1. Performance as the user sees it

| Measurement | Result |
|---|---|
| Landing page load (3 fresh loads, `load` event) | 0.66 s, 0.24 s, 0.37 s (DOMContentLoaded 0.35 s) |
| Sign in → app visible | 0.91 s |
| Ask a question → answer shown ("Who wrote Knowledge Graphs: Opportunities and Challenges?") | 2.42 s |
| Explore / Connect / Research 360 (API time, smoke test) | 1.6-1.7 s each |

## 2. Task checks

| ID | Task | Expected | Observed | Result |
|---|---|---|---|---|
| U01 | Sign in with a wrong password | Clear, non-revealing message | "Incorrect email or password." | Pass |
| U02 | Sign in with the right password | App opens without a page reload | App visible in 0.91 s | Pass |
| U03 | Ask a question | Short answer with clickable citations | Answer with `[doc]` citation links; "Retrieved graph facts" expandable | Pass |
| U04 | Submit an empty question | No error, no empty answer | Answer area stays empty, no error shown | Pass |
| U05 | Explore an entity | Interactive graph with colour legend | Cytoscape graph, 9-type legend, labels on nodes | Pass |
| U06 | Connect two entities | Paths listed and drawn | Paths with citations and a path graph | Pass, see finding F2 |
| U07 | Non-member opens the app | Upload hidden, reason shown | "Uploading papers is limited to project members. You can ask questions and explore everything." | Pass |
| U08 | Keyboard only on the sign-in form | All controls reachable with Tab in a sensible order | Tab moves password → Show → Keep me signed in → Forgot password → Sign in | Pass |
| U09 | Phone width (390 x 844) | No horizontal scrolling, readable layout | No horizontal overflow on the landing page or the app; panels stack in one column | Pass, see finding F3 |

## 3. Accessibility (axe-core 4.10, WCAG 2.0/2.1 A and AA rules)

| Page | Violations |
|---|---|
| Landing page with sign-in form | 1: the password **Show** button had a contrast ratio of 4.4:1 against the field (minimum 4.5:1) |
| Signed-in app | none |

## 4. Findings and fixes

| ID | Finding | Severity | Status |
|---|---|---|---|
| F1 | Contrast of the password **Show** button 4.4:1 (WCAG AA needs 4.5:1) | Low | **Fixed** in `src/frontend/style.css` (now 5.27:1); live after the next Amplify deploy |
| F2 | Connect listed the same path several times when parallel edges linked the same nodes | Medium | **Fixed** in `src/backend/kg/insights.py` (each path shown once); live after the next `sam deploy` |
| F3 | At phone width the empty graph panel fills a whole screen before the first query | Low | Open: collapse the panel until there is something to draw |
| F4 | Sample papers in `dataset/sample_papers.jsonl` had placeholder authors and venues, so an answer named "Sample Author C" as the author of a real paper | High (trust) | **Fixed** in the repository (real authors and venues); the live graph still holds the old placeholder nodes until they are removed and the sample file is re-ingested |
