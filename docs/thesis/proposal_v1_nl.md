# Thesis proposal v1 — Dutch draft (Overleaf-ready)

> **Canonical location:** `C:\Users\joepw\thesis-knowledge\05-drafts\proposal_v1_nl\`
> This file is a pointer; the proposal lives in three LaTeX-ready files in the knowledge base.

## Why this exists

Paul Bouman bevestigde via TMS dat de eerder ingediende TAR weliswaar **goedgekeurd is** (joep mag aan thesis beginnen), maar **niet als proposal volstaat**. Een aparte proposal is de verplichte eerste fase per ESE Econometrie procedure. Paul heeft TMS-undo aangevraagd voor de TAR-als-proposal upload; zodra die landt moet er een echte proposal staan klaar.

## What's in the canonical folder

| File | What |
|---|---|
| `main.tex` | LaTeX-bron, 5 paragrafen, **1095 NL woorden** (target 800-1100). Article-class, 12pt, 1.5 line spacing, 2.5cm marges. natbib + apalike voor APA citations. |
| `references.bib` | 4 verified BibTeX-entries: Markowitz 1952, Steuer & Na 2003, Hirschberger et al. 2013, Pedersen et al. 2021. |
| `README.md` | Overleaf-import-instructies + AI-disclosure-tekst + ESE-checklist-mapping + "Engelse vaktermen niet vertalen"-lijst. |
| `_verify.py` | Eenmalig verificatie-script (woordtelling, banned-elements scan, citation cross-check). Niet bedoeld om te committen — local tool. |

## Workflow

1. Joep opent de drie files in Overleaf (`README.md` legt het uit).
2. Switch `babel` van `dutch` naar `english`, vertaal de 5 paragrafen zelf.
3. Engelse vaktermen blijven Engels (lijst staat in `README.md`).
4. Compileer in Overleaf → PDF download.
5. Wacht tot Paul's TMS-undo binnen is.
6. Upload PDF in TMS phase 3 + planning .xlsx in TMS chat + AI-disclosure-melding in TMS chat.

## Verification snapshot (2026-05-28)

- 5 paragrafen ✅
- 1095 woorden totaal ✅
- 0 banned LaTeX-elementen (geen `\section`, `\subsection`, `\begin{itemize}`, `\begin{equation}`, `\begin{figure}`, `\includegraphics`, `\appendix`) ✅
- 4 citaten in body match 1-op-1 met 4 entries in `references.bib` ✅
- §3 eindigt met expliciete "Hoe..."-vraag ✅
- Alle 6 inhoudelijke ESE-checklist-elementen gedekt (manual §c regels 507-519); (7) planning gaat apart per ESE-procedure ✅

## Sources used (read-only)

- `Desktop/PwC/TAR/thesis_approval_draft.md` (NL TAR)
- `thesis-knowledge/02-literature/portfolio_optimization_v1.md` (verified citations log)
- `docs/thesis/w1_kickoff_brief.md` (research-question concepts)
- `docs/thesis/w1_optimization_methods.md` (methods shortlist rationale)
- `thesis-knowledge/04-experiments/exp01-slsqp-vs-grid-beerwiser.md` (W2 empirisch resultaat → "voorlopig experiment" in §3 en §5)
- `thesis-knowledge/00-meta/writing-a-thesis-eur2026/md/Proposal-intro.md` + per-paragraph chapters (de Bliek's 5-paragraph spec)
- `thesis-knowledge/00-meta/master-thesis-manual-2025-2026.txt` §c regels 494-540 (ESE inhouds-checklist)
