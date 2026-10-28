# R3/R4 requirements coverage

R3/R4 change presentation and the test model, not the approved Native architecture. R4 adds Mermaid and the authorized public-safe PRD synchronization.

Frozen R2 source SHA256: `e4f620ff3b1bde735829b2dd04edaa36ac42a9984fb37c9928de688b43d66a9f`. The reference contains the entire original source verbatim after its amendment banner.

## User stories

| R2 | R3 location | Preserved |
|---|---|---|
| 1 | User Stories / US01 | yes |
| 2 | User Stories / US02 | yes |
| 3 | User Stories / US03 | yes |
| 4 | User Stories / US04 | yes |
| 5 | User Stories / US05 | yes |
| 6 | User Stories / US06 | yes |
| 7 | User Stories / US07 | yes |
| 8 | User Stories / US08 | yes |
| 9 | User Stories / US09 | yes |
| 10 | User Stories / US10 | yes |
| 11 | User Stories / US11 | yes |
| 12 | User Stories / US12 | yes |
| 13 | User Stories / US13 | yes |
| 14 | User Stories / US14 | yes |
| 15 | User Stories / US15 | yes |
| 16 | User Stories / US16 | yes |
| 17 | User Stories / US17 | yes |
| 18 | User Stories / US18 | yes |
| 19 | User Stories / US19 | yes |
| 20 | User Stories / US20 | yes |
| 21 | User Stories / US21 | yes |
| 22 | User Stories / US22 | yes |
| 23 | User Stories / US23 | yes |
| 24 | User Stories / US24 | yes |

## Acceptance gates

| R2 | R3 location | Preserved |
|---|---|---|
| G0 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G1 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G2 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G3 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G4 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G5 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G6 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G7 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G8 | Testing Decisions / gates; full R2 gate text in reference | yes |
| G9 | Testing Decisions / gates; full R2 gate text in reference | yes |

## Delivery phases

| R2 | R3 location | Preserved |
|---|---|---|
| P0 | Delivery table; detailed task plan retained | yes |
| P1 | Delivery table; detailed task plan retained | yes |
| P2 | Delivery table; detailed task plan retained | yes |
| P3 | Delivery table; detailed task plan retained | yes |
| P4 | Delivery table; detailed task plan retained | yes |
| P5 | Delivery table; detailed task plan retained | yes |
| P6 | Delivery table; detailed task plan retained | yes |
| P7 | Delivery table; detailed task plan retained | yes |
| P8 | Delivery table; detailed task plan retained | yes |

## Full technical coverage

- All 15 original Implementation Decisions, testing seams, change inventory, acceptance detail, dependencies, risks, constraints and source links remain verbatim in the linked R2 reference, in both original languages.
- Main R3 explains artifact/numerics/GPU control, ownership/commit including speculative shadow trial, accepted-prefix clamp and replay from retained base state, bridge bounds, session/canonical history, API/status, offload and evidence in separate readable sections.
- Intentional override: Gyro-M replaces Swift as the test target; no restoration/download of Swift is required. The user selected the existing Gyro-M 262K/5-worker profile; runtime compatibility still requires model execution.
- Clarified status: P0 component tests/builds have run; P1+ and model parity/latency gates have not. No microbenchmark is called Native arithmetic evidence.
- R3 was a local reading draft. The user subsequently authorized PRD synchronization. R4 has Thai reading sections plus English requirements, and a public bilingual R2 technical companion; local absolute paths, machine identifiers and unnecessary inventory were excluded.

## kien-thai audit

- Actual skill: `C:/Users/xenod/.agents/skills/kien-thai/SKILL.md`; register: technical explainer, direct factual voice, no first-person narrator and no polite body particles.
- Applied topic-first structure, fronted conditions, paragraph breaks, verbs over nominal padding, consistent developer terms in Latin script, and no recap closing.
- Native-model availability branch ran via its actual Python --check script (the Bash wrapper only delegates to it); localhost:11434 refused the connection. No service/model was started or downloaded; direct drafting fallback was used.
- Main draft was checked against the skill blocklist; no forbidden phrase was used. Technical reference is retained source material, not claimed as rewritten kien-thai prose.

Parent-review correction: readable R3 now states the shadow trial/clamp/replay mechanism directly and distinguishes the 10 successful probe cases from the final 9-test suite (p0-final-suite.log:9 tests,3.814s,OK,DONE0).

Independent R3 review additions: explicit normative R2 precedence with full companion-file hash and four enumerated overrides; chosen262K/RoPE/5workers/config/input equality for controlled arms; exact tokenizer-content/special-ID binding distinguished from geometry/vocabulary size, artifact identity and TQ2_T/MTP runtime proof; Q2 regressions retained without claiming Gyro correctness; build/probe separated from inference/ownership/resources/performance readiness.

## R4 publication verification

- Existing issue: https://github.com/xenodeve/Strata-xeno/issues/276; no duplicate issue.
- Normative bilingual companion: https://github.com/xenodeve/Strata-xeno/issues/276#issuecomment-6479163682
- Appendix payload SHA256:e2b47234a3767b95eb838c7380d5f6d9d4d657f6e460bac4a4276edf5de54204. Exact original EN/TH Implementation Decisions, Testing Decisions, Acceptance Gates and Delivery Dependencies were verified in it. Only obsolete/non-contractual local inventory and workflow status were removed.
- R4 body exact match to both local public-facing PRD files; SHA256:1dcf885081d171b87e8f07026877a3a81bd9ca713e492bcd8bdf1126515f8dba.
- Three diagrams match validated source exactly. Mermaid11.16.0 parser and local headlessChrome render passed; rendered images inspected. P6->P7 dependency retained. Actual GitHub-rendered SVGs were not verified.
- Local report/model/profiles remain private; no repository push/PR/merge/deployment.
