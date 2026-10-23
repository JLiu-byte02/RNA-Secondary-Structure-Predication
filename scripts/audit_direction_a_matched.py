"""Canonical audit entry for frozen matched-cache source (do not rerun inference)."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.direction_a_protocol import ROOT,write_json,file_sha
from scripts.direction_a_audit_rows import normalizing_metric_row,verify_dependencies
from scripts import run_direction_a_matched as frozen

def main():
    cfg=ROOT/'configs/direction_a_matched_audit.json';manifest=json.loads(cfg.read_text())
    verify_dependencies(ROOT,manifest['files'])
    # The cache-producing file remains byte-identical to its execution snapshot.
    # Only decoded JSON lists are adapted to the original helper's set-of-tuples API.
    frozen.metric_row=normalizing_metric_row(frozen.metric_row)
    frozen.report()
    verify_dependencies(ROOT,manifest['files'])
    path=ROOT/'reports/direction_a_stage2_2026-10-23/matched_verification.json';row=json.loads(path.read_text())
    row['audit_binding']=dict(configuration_sha256=file_sha(cfg),source_snapshot=manifest['source_snapshot'],files=manifest['files'],
        audit_entry_sha256=file_sha(Path(__file__)),normalization_adapter_sha256=file_sha(ROOT/'scripts/direction_a_audit_rows.py'),
        scope='Supplemental executable dependency binding verified against the immutable pre-execution Git snapshot. JSON pair normalization changes audit input types only; cache and selection sources remain byte-identical.')
    write_json(path,row)

if __name__=='__main__':main()
