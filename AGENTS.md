# HC initiation research

- Read README.md first. It is the canonical plan, evidence summary and milestone checklist for this repository.
- Scope is HC initiation during observed NE support. The separate ICU world-model repository and paused NE/VP projects must not be changed or staged here.
- Preserve the frozen MIMIC actual-dose analysis. Order-based analyses have their own eligibility, time zero, risksets and weights. Do not pool incompatible estimands or choose analyses by mortality direction.
- The four files in analysis/steroid_*.py are explicit source exports of existing canonical HC scripts. Edit those canonical files and run sync_sources.py. The export changes path configuration only and asserts other Python syntax is unchanged. New order-cohort work is maintained directly in hc_order_bridge.py.
- Machine paths are in ignored config.local.json or HC_TTE_CONFIG. Raw data, patient derivatives, identifiers, free text, fitted objects, caches, credentials and complete local reports must never be committed, including to this private repository.
- Stage exact allowlisted files. Check the complete staged content before each push. No automatic invitations, public visibility changes or clinical-data uploads.
- Python3.8/CPU; use the installed environment. Do not install or upgrade packages.
- New raw reads require a specific denominator/field question, columns, filters and estimated cost first. Reuse completed projections, do not scan unrelated tables.
- Keep observation time, availability time, order time and administration time distinct. Missing records are not normal values, no treatment, full code or survival.
- Update README checkboxes only after the specified acceptance test passes. No duplicate plans or audit reports; preserve prior conclusions and estimates.
- Use codex/ branches for implementation. Synthetic tests must cover substantive time, selection, linkage and inference contracts; do not refit frozen outcomes for documentation changes.
