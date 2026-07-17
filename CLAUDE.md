# Claude Code Instructions

Read `AGENTS.md` in this directory for full instructions on developing Reachy Mini applications.

---

# Avail Intern fork — repo rules

This is Avail's fork of `pollen-robotics/reachy_mini_conversation_app`: the embodiment
layer for the **Avail Intern**, a Reachy Mini (Wireless, Pi 5) office assistant whose
agentic backend is a Hermes Agent instance on a sandboxed VM (see `profiles/avail_intern/ask_hermes.py`).

## Rules

- **NEVER `git push` without Vibhu's explicit approval.** Commit locally freely, then
  present the changes for review and wait for a yes before pushing. This applies to
  every branch and every remote, every session, no exceptions.
- `origin` = vibhurajeev's fork; `upstream` = pollen-robotics. Sync upstream via
  `git fetch upstream` + merge — never push to upstream.
- Keep divergence from upstream minimal and additive where possible (new files over
  edits to theirs) so upstream merges stay cheap.

## Repo-local git quirks (already configured)

SSH on this machine is Vibhu's personal access — Claude must not use or work around
it. Ask Vibhu to run any SSH operation himself (VM, robot, git-over-SSH). For git in
this repo, Claude uses HTTPS: the global gitconfig rewrites GitHub HTTPS → SSH, so
this repo has local identity `insteadOf` rules for `vibhurajeev/` and
`pollen-robotics/` plus `credential.helper = !gh auth git-credential`. Don't switch
the remotes to SSH.

## Layout of our additions

- `profiles/avail_intern/` — the Avail Intern personality, tool list, and the
  `ask_hermes` SSH bridge tool to the Hermes VM.
- `INSTALL.md` — end-to-end install guide (robot + VM + Slack).
- `ARCHITECTURE.md` — map of how the upstream app works; read it before changing core code.

## Roadmap

Phase 2 is an ambient vision loop (react to people/objects unprompted) — the reason
this fork exists. Design it as new modules + minimal hooks, per the divergence rule.
