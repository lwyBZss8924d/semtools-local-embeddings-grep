# Semtools AIHT Skills

`skills/semtools-skills-collection/` is the repository-owned maintenance source for the custom semtools AIHT skills.

The global runtime projection lives at `~/.agents/skills/semtools-skills-collection/`. Keep this repository copy authoritative for fork-specific semtools behavior, especially the local-source `semtools jgrep` Jina semantic grep/classification surface. Runtime projection or installation tooling can copy from this directory into the global skills tree.

The copied collection intentionally excludes local OS metadata such as `.DS_Store`.
