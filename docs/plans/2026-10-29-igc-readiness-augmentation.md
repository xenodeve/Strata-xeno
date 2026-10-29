# Prospective matched-readiness training coverage

The original MLP misses one equal-operand subtraction; the linear control instead
admits unsupported division. Both fit all training rows. These are development
findings, not proof of a quantization or architecture defect. Test additional
training coverage with the original MLP/optimizer and unchanged development rubric.

Add416 fit-only rows, seed20261108:192 new operand pairs (64 per operation), each
with calculation and copy/quotation intents, plus32 unsupported-operation negatives.
Per operation eight pairs have equal operands and eight have one zero operand, selected from
unused pairs; the rest are ordinary signed magnitude≤99. Add new fit templates
without using old development templates/pairs. Exclude all previous input/output
and fresh-acceptance64 pairs. Explicit unsupported examples span division, roots,
transcendentals, powers, variables and decimal operands; they are outside this
initial arithmetic module, not excluded from the long-term project.

Collect only the416 new final-anchor vectors with the same qualified actual-Hub
collector/lifecycle/profile and source identity as the original320. No expected
answer, execution result or generated-token feature. Retain original capture and
fit files. Bind each capture to its own immutable protocol and reconstruct every
record independently before concatenation. New fit624 rows, unchanged development
112. No mixing of development into normalization, fitting or checkpoint choice.

Use the original328,066-parameter2560→128→2 GELU MLP, same seed20261107,300
full-batch AdamW steps lr0.001/weight_decay0.01, deterministic CPU F32/four threads,
120s fitting bound and final argmax. Only data/fit normalization population changes;
no threshold/architecture adjustment. All112 known development rows must classify
correctly for this diagnostic to pass. Record any remaining missed/false calls.
This is development-guided training, not a fresh generalization claim; acceptance
64 remains untouched until operand extraction and the other eligibility gates pass.

Before capture, corpus preparation exposed only15 unused lhs-zero pairs for the
24-row quota, but19 more rhs-zero pairs. Select either orientation without reusing
any reserved pair. This fixes a data-preparation exhaustion error, not the model;
the strict disjointness and total zero/equality quotas remain unchanged.
