# WorkflowEvals and its evaluation datasets

Recorded 2026-11-10 for PR #83. The integration was introduced at `9c57e86`; this record is retrospective.

WorkflowEvals code is Apache-2.0 at [0ac3b8a](https://github.com/typesafe-ai/WorkflowEvals/blob/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb/LICENSE). It is fetched into an ignored checkout; only a connectivity/CLI patch is shipped here. Upstream scoring and workflow logic remain unchanged.

| Component | Pinned revision | License evidence | Decision |
|---|---|---|---|
| typesafe/evalsafe-invoice-processing | `6beeb2d2acd65c086c835022f5f4d7434114cafc` | [Apache-2.0 LICENSE](https://huggingface.co/datasets/typesafe/evalsafe-invoice-processing/blob/6beeb2d2acd65c086c835022f5f4d7434114cafc/LICENSE) | Adopt |
| typesafe/evalsafe-customer-service | `b1342f5a704587dbc465867c38c2694348ff86e4` | [No license in files/card/manifest](https://huggingface.co/datasets/typesafe/evalsafe-customer-service/tree/b1342f5a704587dbc465867c38c2694348ff86e4) | Pending written upstream license; runs blocked |
| typesafe/evalsafe-security-incidents | `fbe1ea5c69cf494157fd23f2002a0d9d9a418443` | [No license in files/card/manifest](https://huggingface.co/datasets/typesafe/evalsafe-security-incidents/tree/fbe1ea5c69cf494157fd23f2002a0d9d9a418443) | Pending written upstream license; runs blocked |
| typesafe/evalsafe-agent-trace-observability | `8635540973910a92465fe2bc53e195375aa6e1a8` | [No license in files/card/manifest](https://huggingface.co/datasets/typesafe/evalsafe-agent-trace-observability/tree/8635540973910a92465fe2bc53e195375aa6e1a8) | Pending written upstream license; runs blocked |

## Evaluation and conditions

We need the original vendor workflow inputs, policies and saved reference actions. Reimplementing the benchmark would introduce scoring drift; Kev and Decision Index test different tasks. The pinned upstream is small, has locked dependencies, and can be removed without affecting serving or training.

Code/invoice score: functional fit 19/20; license fit 20/20; maintenance 14/20 (small vendor-owned project); security 12/20 (locked dependencies, no security audit); integration/exit cost 18/20. Total 83/100. Adopt the code and invoice dataset with attribution, original license text in the fetched checkout/dataset cache, and pinned revisions. Re-evaluate on 2027-11-10 or any license/security/upstream change.

The other three datasets have no LICENSE file, license metadata, or license declaration in dataset.json at either the pinned revision or current main (checked 2026-11-10). Do not infer their license from the code or invoice dataset. No score/adoption is assigned; downloads/runs through this integration are blocked until an explicit upstream license statement is linked and reviewed. Previously executed pilots predate this intake check. No dataset content is committed here.

The score assesses this optional benchmark integration, not the validity of model-generated reference labels. Vulnerability review is not claimed. Setup records must be updated if the upstream patch or dataset pins change.
