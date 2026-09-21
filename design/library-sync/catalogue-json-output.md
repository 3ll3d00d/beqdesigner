# BEQCatalogue JSON output migration

> The output contract is owned by
> [`beqcatalogue/docs/filter-record-contract.md`](../../../beqcatalogue/docs/filter-record-contract.md).

The library publisher replaces its XML filter output with one version-1 BEQ
filter-record JSON object per title and an aggregate `database.json` in the
filter repository. The images repository remains unchanged. Source records
carry the persisted portion of a final database object; BEQCatalogue adds its
page URL and other derived fields while building the public global catalogue.
They are not BEQDesigner’s internal filter/project JSON.

Implementation must replace XML rendering, path naming, repository scanning,
digest/revision/reopen handling, settings wording and CLI options together.
BEQCatalogue must ingest individual record files and aggregate them before its
normal page/database generation. Tests must round-trip a BEQDesigner-published
record through BEQCatalogue and `model.catalogue.CatalogueEntry`.

## Status

The publisher, filter-repo path naming, derived aggregate and local catalogue
identity scan now use JSON records. The serializer is tested against the
BEQCatalogue 96 kHz coefficient convention, including shelf unrolling.
Compatibility names such as `xml_repo` and `xml_dir` remain internal plumbing
for this first landing; their settings, CLI and UI wording still need the
separate migration described above. The BEQCatalogue reader is available for
onboarding a record repository through `record_repo_configs`.
