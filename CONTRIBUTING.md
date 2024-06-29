# Contributing to feotensor

## Commit Convention

`feotensor` infrastructure enforces [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).
A `commitlint` CI check ensures that commits to `main` are all of this format. We use these commits
to automatically generate a release number according to compatible versioning (ComVer).

ComVer allows only major or minor version bumps (no patch). A major version bump is not backwards
compatible. A minor version bump is backwards compatible.

Major version bumps are achieved either by adding a `!` immediately before the `:`, or by adding
`BREAKING CHANGE` to the commit footer.

Otherwise all other commits result either in no release, or in a minor bump. For details on exactly
which commit types can trigger a release, and which commit types cannot, see `release.config.js`.
