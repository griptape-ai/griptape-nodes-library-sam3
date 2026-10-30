## Changelog

`CHANGELOG.md` follows [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/). Each version is the `library_version` in `griptape_nodes_sam3_library/griptape-nodes-library.json`. Its readers are artists using these nodes and the agents that build workflows with them.

**Add an entry for every user-facing change** - In the same PR, add a file to `changelog.d/` holding the entry. User-facing means anything a user notices after updating the library: a node added, removed, or renamed; a parameter, default, or output that differs; a provider model added or retired; a saved workflow that loads or runs differently. After adding one, tell the user so they can review the wording.

**One file per entry, named for its type** - Never edit `## [Unreleased]` in `CHANGELOG.md` directly; `make check/changelog` rejects it. Create `changelog.d/<type>-<slug>.md`, where `<type>` is one of `added`, `changed`, `deprecated`, `removed`, `fixed`, or `security`, and `<slug>` is a few lowercase words joined by hyphens. The whole file is the bullet, with no heading. `make changelog` prints `[Unreleased]` with these files folded in. `changelog.d/README.md` is the same contract for humans.

**Skip what users never see** - Refactors, tests, CI, docs-only changes, and dev dependency bumps get no entry. Neither does a fix for a bug that never shipped in a version: edit or delete the entry that introduced it.

**Describe the result, not the code** - Name the node and parameter in backticks or by their editor label, say what the user sees now, present tense, one or two sentences. `Fixed` if the old behavior was a bug, `Changed` if it was intentional. Link the GitHub issue on its own last line. Leave out PR numbers and `@handles`. Wrap at about 100 characters, continuation lines indented two spaces.

**Mark breaking changes** - A change that makes a saved workflow stop working, or give a different result, without edits starts with `**Breaking:**`. Say what breaks and what to do about it.

**Leave `CHANGELOG.md` to the version bump** - Do not add version headings or link definitions. `make version/patch`, `make version/minor`, `make version/major`, and `make version/set` fold the `changelog.d/` entries into the new version section and delete them. Fixing a mistake in a released entry is fine.
