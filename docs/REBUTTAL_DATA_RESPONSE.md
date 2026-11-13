# Copy-ready response on data construction and trajectory validity

Thank you for raising the concern about the construction and validity of the positive/negative
references. We agree that our earlier wording was too strong. In particular, the public AgentHarm
and Agent-SafetyBench releases used in this analysis contain task records rather than natively paired,
executed safe/unsafe trajectories. We have corrected the manuscript and no longer describe these
records as counterfactual trajectory pairs.

We added a source-level audit against pinned official repositories. Every embedded source object in
our artifact exactly matches the corresponding official JSON object (674/674 records), with no prompt,
environment parameter, action, or observation rewritten. AgentHarm provides 208 native benign/harmful
task pairs joined by the official behavior ID (176 public-test and 32 validation pairs). Of these, 171
pairs have exactly identical `target_functions` arrays. Because the AgentHarm dataset card explicitly
requests evaluation-only use, all AgentHarm records are now excluded from optimization, teacher
conditioning, reference construction, and hyperparameter selection.

Agent-SafetyBench provides no native positive/negative pair key. We therefore treat its 129 matched
pairs only as a task-selection and coverage index. A positive-side record has `fulfillable=1`, a
negative-side record has `fulfillable=0`, and every selected edge shares at least one exact risk label,
environment name, and tool name. Each official record appears at most once. We independently recomputed
the full bipartite graph over all 735 fulfillable and 1,265 unfulfillable records. Hopcroft-Karp and
NetworkX both return a maximum cardinality of 129, confirming that the selected 129 edges constitute a
maximum-cardinality matching under the stated graph. Relaxing the edge definition gives maxima of 187
(same environment and shared tool), 297 (shared tool), 435 (shared environment), and 699 (shared risk).

Crucially, neither the native AgentHarm pairs nor the matched Agent-SafetyBench records are fed directly
to HFDTree as `c+`/`c-`. We now require a fail-closed execution protocol. For every training task, we
restore the same initial environment checkpoint for each branch, record the complete action-observation
sequence, and run a named and versioned environment scorer. A positive reference is admitted only when
the executed rollout is both safe and task-successful; a negative reference must be an actually executed
violation from the same task and initial-state fingerprint. Tasks for which either outcome is unavailable
are retained only for coverage accounting and never completed with a synthetic trajectory. The released
validator rejects task-level pseudo-pairs, different initial states, missing action-observation traces,
and unverified scorer labels.

We also separate two evaluation tracks. The primary cross-benchmark track keeps every AgentHarm and
Agent-SafetyBench task evaluation-only and trains on disjoint environments. A separately labeled
in-domain diagnostic uses a deterministic 103/26 split of the 129 Agent-SafetyBench pair groups (206/52
task seeds), with no source-record overlap. We do not present this diagnostic as untouched official
benchmark generalization. For both tracks we will report the number and eligibility rate of genuinely
executed reference pairs, per-risk results, three training seeds, task-cluster bootstrap confidence
intervals, and paired tests against the strongest baseline.

These changes narrow the claim to what the public data and executed evidence actually support, prevent
benchmark contamination, and make the distinction between task similarity and same-state trajectory
counterfactuals explicit and machine-checkable.
