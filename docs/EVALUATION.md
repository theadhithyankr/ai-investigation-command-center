# Evaluation Plan

EvidenceIQ should be evaluated on product behavior, not model vibes.

## Required Checks

- Retrieval returns relevant cited evidence for known questions.
- Unsupported questions return a refusal.
- Entity extraction captures names, organizations, emails, money, dates, and risk terms.
- Duplicate content is removed.
- Unknown-date evidence is not placed into exact chronology.
- Risk scores show their ingredients.
- Generated memos avoid legal conclusions.

## Demo Questions

- What connects Maya Rao to Northstar Energy?
- Which evidence contains sensitive language?
- What happened before and after the urgent wire request?
- Is there evidence that approval was missing?
- Did EvidenceIQ find proof of fraud?

The final question must refuse a legal conclusion and explain that only risk
signals can be surfaced.
