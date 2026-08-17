# M6 Controlled-Error False Reject Audit

The five False Rejects are fully explained controlled extractor omissions; no StateTracker, AP grounding, action alignment, or DFA implementation error was found.

- Audit digest: `59a582d31396b548c0aa2c9dfc78cb5c93f6d6347a8e073d1ce0d5f291648032`
- Audit schema: `agemem.m6_false_reject_audit.v2`
- M6 report digest: `e803f7752dc9e7357284887cf7716273bbd5396f62db1fc438d7cad95a2f9f92`
- DFA/reward action checks: `74`
- Source lineage: M5/M6/config/manifest digests, report byte SHA, per-file hashes, row counts, and action coordinates verified
- Human-backed FA/FR: `0/20`, `0/10`
- Controlled-error FA/FR: `0/20`, `5/10`
- M7 entry gate: **PASS**
- Real LLM calls: `0`

| task / rollout | dropped fact | missing Triple | first action | reward | classification |
|---|---|---|---|---:|---|
| `hotpot-5a74b19355429916b01641dd` / `m5-train-5a74b19355429916b01641dd-gold` | `hp-5a74b19355429916b01641dd-2fe86f28b15b6b5b` | Sunye / birth_date / 1989-08-12 | `m5-train-5a74b19355429916b01641dd-gold:call:0` | 2.00 -> 1.25 (-0.75) | `expected_extractor_omission` |
| `hotpot-5a83df2655429933447460a1` / `m5-train-5a83df2655429933447460a1-gold` | `hp-5a83df2655429933447460a1-8260aac478284146` | Grant O'Riley / played_for / Fitzroy Football Club | `m5-train-5a83df2655429933447460a1-gold:call:0` | 2.00 -> 1.25 (-0.75) | `expected_extractor_omission` |
| `hotpot-5a85aaee5542991dd0999e84` / `m5-train-5a85aaee5542991dd0999e84-gold` | `hp-5a85aaee5542991dd0999e84-0b13cfadf6b836ef` | Junko Noda / known_for / Love Hina | `m5-train-5a85aaee5542991dd0999e84-gold:call:0` | 2.00 -> 1.25 (-0.75) | `expected_extractor_omission` |
| `hotpot-5a8ac7d055429950cd6afb8f` / `m5-train-5a8ac7d055429950cd6afb8f-gold` | `hp-5a8ac7d055429950cd6afb8f-a72b7f3ef47c9649` | Veitchia / plant_family / Arecaceae | `m5-train-5a8ac7d055429950cd6afb8f-gold:call:0` | 2.00 -> 1.25 (-0.75) | `expected_extractor_omission` |
| `hotpot-5abecbed5542997719eab5c5` / `m5-train-5abecbed5542997719eab5c5-gold` | `hp-5abecbed5542997719eab5c5-1f0bc4a7ff5f3cda` | Coming of Age / cancelled_with / Ideal | `m5-train-5abecbed5542997719eab5c5-gold:call:0` | 2.00 -> 1.25 (-0.75) | `expected_extractor_omission` |

## Per-trajectory diagnosis

### `hotpot-5a74b19355429916b01641dd` / `m5-train-5a74b19355429916b01641dd-gold`

- Injection: `drop_relevant_fact` on `hp-5a74b19355429916b01641dd-2fe86f28b15b6b5b`.
- Missing Triple: `Sunye / birth_date / 1989-08-12`.
- First difference: `m5-train-5a74b19355429916b01641dd-gold:call:0` at timestep `0`.
- Oracle AP: `['observed_supporting_fact', 'stored_supporting_fact']`; extracted AP: `['observed_supporting_fact']`.
- StateTracker: `correct=True`; missing StateFact IDs: `['c2ecf7e70faa209a98a415dbe2f51411ea1fb063110ffea7f82eed52221060f2']`.
- AP grounding: `correct=True`; action alignment: `correct=True`.
- First DFA divergence: Oracle `q0 -> q1` via `['progress_store_support']`; extracted `q0 -> q0` via `[]`.
- DFA implementation: `correct=True`; definition too strict: `False`.
- Classification: `expected_extractor_omission`; causal chain complete: `True`.
- Differing action chain:

  - `m5-train-5a74b19355429916b01641dd-gold:call:0`: AP oracle/extracted `['observed_supporting_fact', 'stored_supporting_fact']` / `['observed_supporting_fact']`; DFA `q0->q1` / `q0->q0`; reward error `-0.25`.
  - `m5-train-5a74b19355429916b01641dd-gold:call:1`: AP oracle/extracted `['stored_supporting_fact']` / `['stored_supporting_fact']`; DFA `q1->q1` / `q0->q1`; reward error `+0.25`.
  - `m5-train-5a74b19355429916b01641dd-gold:call:4`: AP oracle/extracted `['retrieved_supporting_fact']` / `[]`; DFA `q1->q1` / `q1->q1`; reward error `+0.00`.
  - `m5-train-5a74b19355429916b01641dd-gold:call:5`: AP oracle/extracted `['retrieved_supporting_fact', 'supporting_coverage_complete']` / `['retrieved_supporting_fact']`; DFA `q1->q3` / `q1->q1`; reward error `-0.50`.
  - `m5-train-5a74b19355429916b01641dd-gold:call:6`: AP oracle/extracted `['supporting_coverage_complete', 'answered_correctly']` / `['answered_correctly']`; DFA `q3->q4` / `q1->q_reject`; reward error `-0.25`.

### `hotpot-5a83df2655429933447460a1` / `m5-train-5a83df2655429933447460a1-gold`

- Injection: `drop_relevant_fact` on `hp-5a83df2655429933447460a1-8260aac478284146`.
- Missing Triple: `Grant O'Riley / played_for / Fitzroy Football Club`.
- First difference: `m5-train-5a83df2655429933447460a1-gold:call:0` at timestep `0`.
- Oracle AP: `['observed_supporting_fact', 'stored_supporting_fact']`; extracted AP: `['observed_supporting_fact']`.
- StateTracker: `correct=True`; missing StateFact IDs: `['abc4e2bf891abd66d3624c8fec1127c6a061afe6eb0497a037d4667a0ff2a62a']`.
- AP grounding: `correct=True`; action alignment: `correct=True`.
- First DFA divergence: Oracle `q0 -> q1` via `['progress_store_support']`; extracted `q0 -> q0` via `[]`.
- DFA implementation: `correct=True`; definition too strict: `False`.
- Classification: `expected_extractor_omission`; causal chain complete: `True`.
- Differing action chain:

  - `m5-train-5a83df2655429933447460a1-gold:call:0`: AP oracle/extracted `['observed_supporting_fact', 'stored_supporting_fact']` / `['observed_supporting_fact']`; DFA `q0->q1` / `q0->q0`; reward error `-0.25`.
  - `m5-train-5a83df2655429933447460a1-gold:call:1`: AP oracle/extracted `['stored_supporting_fact']` / `['stored_supporting_fact']`; DFA `q1->q1` / `q0->q1`; reward error `+0.25`.
  - `m5-train-5a83df2655429933447460a1-gold:call:4`: AP oracle/extracted `['retrieved_supporting_fact']` / `[]`; DFA `q1->q1` / `q1->q1`; reward error `+0.00`.
  - `m5-train-5a83df2655429933447460a1-gold:call:5`: AP oracle/extracted `['retrieved_supporting_fact', 'supporting_coverage_complete']` / `['retrieved_supporting_fact']`; DFA `q1->q3` / `q1->q1`; reward error `-0.50`.
  - `m5-train-5a83df2655429933447460a1-gold:call:6`: AP oracle/extracted `['supporting_coverage_complete', 'answered_correctly']` / `['answered_correctly']`; DFA `q3->q4` / `q1->q_reject`; reward error `-0.25`.

### `hotpot-5a85aaee5542991dd0999e84` / `m5-train-5a85aaee5542991dd0999e84-gold`

- Injection: `drop_relevant_fact` on `hp-5a85aaee5542991dd0999e84-0b13cfadf6b836ef`.
- Missing Triple: `Junko Noda / known_for / Love Hina`.
- First difference: `m5-train-5a85aaee5542991dd0999e84-gold:call:0` at timestep `0`.
- Oracle AP: `['observed_supporting_fact', 'stored_supporting_fact']`; extracted AP: `['observed_supporting_fact']`.
- StateTracker: `correct=True`; missing StateFact IDs: `['dd69306e826369714f1c4aec334a2377c0cd5c9dbb5ed2e9498877ec40425fe3']`.
- AP grounding: `correct=True`; action alignment: `correct=True`.
- First DFA divergence: Oracle `q0 -> q1` via `['progress_store_support']`; extracted `q0 -> q0` via `[]`.
- DFA implementation: `correct=True`; definition too strict: `False`.
- Classification: `expected_extractor_omission`; causal chain complete: `True`.
- Differing action chain:

  - `m5-train-5a85aaee5542991dd0999e84-gold:call:0`: AP oracle/extracted `['observed_supporting_fact', 'stored_supporting_fact']` / `['observed_supporting_fact']`; DFA `q0->q1` / `q0->q0`; reward error `-0.25`.
  - `m5-train-5a85aaee5542991dd0999e84-gold:call:1`: AP oracle/extracted `['stored_supporting_fact']` / `['stored_supporting_fact']`; DFA `q1->q1` / `q0->q1`; reward error `+0.25`.
  - `m5-train-5a85aaee5542991dd0999e84-gold:call:5`: AP oracle/extracted `['retrieved_supporting_fact']` / `[]`; DFA `q1->q1` / `q1->q1`; reward error `+0.00`.
  - `m5-train-5a85aaee5542991dd0999e84-gold:call:7`: AP oracle/extracted `['retrieved_supporting_fact', 'supporting_coverage_complete']` / `['retrieved_supporting_fact']`; DFA `q1->q3` / `q1->q1`; reward error `-0.50`.
  - `m5-train-5a85aaee5542991dd0999e84-gold:call:8`: AP oracle/extracted `['supporting_coverage_complete', 'answered_correctly']` / `['answered_correctly']`; DFA `q3->q4` / `q1->q_reject`; reward error `-0.25`.

### `hotpot-5a8ac7d055429950cd6afb8f` / `m5-train-5a8ac7d055429950cd6afb8f-gold`

- Injection: `drop_relevant_fact` on `hp-5a8ac7d055429950cd6afb8f-a72b7f3ef47c9649`.
- Missing Triple: `Veitchia / plant_family / Arecaceae`.
- First difference: `m5-train-5a8ac7d055429950cd6afb8f-gold:call:0` at timestep `0`.
- Oracle AP: `['observed_supporting_fact', 'stored_supporting_fact']`; extracted AP: `['observed_supporting_fact']`.
- StateTracker: `correct=True`; missing StateFact IDs: `['0fb6ee9b22e815bcac9cae3616bbda75bdc0fc35f13140ed67bb5de414f31f4e']`.
- AP grounding: `correct=True`; action alignment: `correct=True`.
- First DFA divergence: Oracle `q0 -> q1` via `['progress_store_support']`; extracted `q0 -> q0` via `[]`.
- DFA implementation: `correct=True`; definition too strict: `False`.
- Classification: `expected_extractor_omission`; causal chain complete: `True`.
- Differing action chain:

  - `m5-train-5a8ac7d055429950cd6afb8f-gold:call:0`: AP oracle/extracted `['observed_supporting_fact', 'stored_supporting_fact']` / `['observed_supporting_fact']`; DFA `q0->q1` / `q0->q0`; reward error `-0.25`.
  - `m5-train-5a8ac7d055429950cd6afb8f-gold:call:1`: AP oracle/extracted `['stored_supporting_fact']` / `['stored_supporting_fact']`; DFA `q1->q1` / `q0->q1`; reward error `+0.25`.
  - `m5-train-5a8ac7d055429950cd6afb8f-gold:call:4`: AP oracle/extracted `['retrieved_supporting_fact']` / `[]`; DFA `q1->q1` / `q1->q1`; reward error `+0.00`.
  - `m5-train-5a8ac7d055429950cd6afb8f-gold:call:5`: AP oracle/extracted `['retrieved_supporting_fact', 'supporting_coverage_complete']` / `['retrieved_supporting_fact']`; DFA `q1->q3` / `q1->q1`; reward error `-0.50`.
  - `m5-train-5a8ac7d055429950cd6afb8f-gold:call:6`: AP oracle/extracted `['supporting_coverage_complete', 'answered_correctly']` / `['answered_correctly']`; DFA `q3->q4` / `q1->q_reject`; reward error `-0.25`.

### `hotpot-5abecbed5542997719eab5c5` / `m5-train-5abecbed5542997719eab5c5-gold`

- Injection: `drop_relevant_fact` on `hp-5abecbed5542997719eab5c5-1f0bc4a7ff5f3cda`.
- Missing Triple: `Coming of Age / cancelled_with / Ideal`.
- First difference: `m5-train-5abecbed5542997719eab5c5-gold:call:0` at timestep `0`.
- Oracle AP: `['observed_supporting_fact', 'stored_supporting_fact']`; extracted AP: `['observed_supporting_fact']`.
- StateTracker: `correct=True`; missing StateFact IDs: `['b28f8a0dc1db46f66eb07abb194d5cfdaeb475d8bbb8c00e40d53ba17470738f']`.
- AP grounding: `correct=True`; action alignment: `correct=True`.
- First DFA divergence: Oracle `q0 -> q1` via `['progress_store_support']`; extracted `q0 -> q0` via `[]`.
- DFA implementation: `correct=True`; definition too strict: `False`.
- Classification: `expected_extractor_omission`; causal chain complete: `True`.
- Differing action chain:

  - `m5-train-5abecbed5542997719eab5c5-gold:call:0`: AP oracle/extracted `['observed_supporting_fact', 'stored_supporting_fact']` / `['observed_supporting_fact']`; DFA `q0->q1` / `q0->q0`; reward error `-0.25`.
  - `m5-train-5abecbed5542997719eab5c5-gold:call:1`: AP oracle/extracted `['stored_supporting_fact']` / `['stored_supporting_fact']`; DFA `q1->q1` / `q0->q1`; reward error `+0.25`.
  - `m5-train-5abecbed5542997719eab5c5-gold:call:4`: AP oracle/extracted `['retrieved_supporting_fact']` / `[]`; DFA `q1->q1` / `q1->q1`; reward error `+0.00`.
  - `m5-train-5abecbed5542997719eab5c5-gold:call:5`: AP oracle/extracted `['retrieved_supporting_fact', 'supporting_coverage_complete']` / `['retrieved_supporting_fact']`; DFA `q1->q3` / `q1->q1`; reward error `-0.50`.
  - `m5-train-5abecbed5542997719eab5c5-gold:call:6`: AP oracle/extracted `['supporting_coverage_complete', 'answered_correctly']` / `['answered_correctly']`; DFA `q3->q4` / `q1->q_reject`; reward error `-0.25`.

## Verified causal chain

`relevant Triple dropped -> corresponding semantic evidence absent -> stored/retrieved AP absent -> coverage fails closed -> DFA stays q1 -> correct terminal answer is rejected`

Both configured value corruptions target irrelevant facts. They affect Triple scoring but cause no AP false positive, False Accept, or False Reject.
