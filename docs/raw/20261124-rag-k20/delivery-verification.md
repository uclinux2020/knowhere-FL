# Delivery verification (2026-11-24)

This Knowhere change is documentation/evidence only. No library, kernel or build
configuration changed; no C++ rebuild or hardware run was performed for delivery.

- `git diff --cached --check`: passed.
- All staged JSON/JSONL parsed successfully.
- Exactly 60 scored rows cover rounds 0..2 × queries 0..19, plus 3 separate warmups.
- Every recorded result reports ordered-ID identity; `stages.json` request records
  exactly match the raw JSONL, and the reported mean matches the raw samples.
- Relative raw-data link exists; four cross-repository links were checked against
  FlagFit's staged tree (the two repositories are delivered together).
- `pre-commit run --files ...` was **not run**: the execution permission check
  refused hooks from the external repository. No alternate hook invocation or
  permission bypass was attempted. `pre-commit` was also not found on PATH.
- High-k AOT execution and private-network Milvus service checks remain blocked as
  recorded in the report. Publication authorization is not treated as permission
  to run those steps.

The existing untracked `.codegraph/` directory is excluded. Source vectors,
credentials, service state and binaries are not part of this commit.
