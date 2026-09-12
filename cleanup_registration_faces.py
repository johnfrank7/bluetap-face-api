"""Safe maintenance preview for temporary registration-face entries.

This script never deletes finalized user faces. With no flags it is a dry run.
"""
import argparse
import json

from face_service import FACE_DATABASE_PATH, _now, _parse_timestamp


def main():
    parser = argparse.ArgumentParser(description="Preview safe temporary face cleanup")
    parser.add_argument("--apply-expired", action="store_true",
                        help="remove only expired temporary registration records")
    parser.add_argument("--orphan-subjects", metavar="JSON_FILE",
                        help="reviewed JSON array of legacy subject IDs confirmed to be abandoned")
    parser.add_argument("--apply-orphans", action="store_true",
                        help="delete only reviewed legacy IDs from --orphan-subjects; never finalized records")
    args = parser.parse_args()
    if args.apply_orphans and not args.orphan_subjects:
        parser.error("--apply-orphans requires --orphan-subjects")

    # Read raw JSON rather than the runtime loader: a dry run must still report
    # legacy databases that the active recognition service rightly rejects.
    if not FACE_DATABASE_PATH.exists():
        print("No face database exists; nothing to clean.")
        return
    database = json.loads(FACE_DATABASE_PATH.read_text(encoding="utf-8"))
    temporary = database.get("temporary_registrations", [])
    legacy = [subject for subject in database.get("subjects", []) if subject.get("finalized") is not True]
    reviewed_orphans = set()
    if args.orphan_subjects:
        reviewed = json.loads(open(args.orphan_subjects, encoding="utf-8").read())
        if not isinstance(reviewed, list) or not all(isinstance(value, str) for value in reviewed):
            parser.error("--orphan-subjects must contain a JSON array of subject IDs")
        reviewed_orphans = set(reviewed)
    total = len(temporary)
    now = _now()
    retained = [record for record in temporary if (expires_at := _parse_timestamp(record.get("expires_at"))) and expires_at > now]
    expired = total - len(retained)

    print(f"Temporary registration records: {total}")
    print(f"Expired temporary records eligible for cleanup: {expired}")
    print(f"Finalized enrollments protected from cleanup: {sum(subject.get('finalized') is True for subject in database.get('subjects', []))}")
    if legacy:
        print(f"Legacy unclassified subject records requiring manual reconciliation: {len(legacy)}")
        print("They are not searched as finalized and this tool will not delete them.")
    confirmed_orphans = [subject for subject in legacy if subject.get("subject_id") in reviewed_orphans]
    if reviewed_orphans:
        print(f"Reviewed legacy orphan records eligible for removal: {len(confirmed_orphans)}")

    changed = False
    if args.apply_expired and expired:
        database["temporary_registrations"] = retained
        changed = True
    if args.apply_orphans and confirmed_orphans:
        database["subjects"] = [
            subject for subject in database.get("subjects", [])
            if subject.get("finalized") is True or subject.get("subject_id") not in reviewed_orphans
        ]
        changed = True
    if changed:
        FACE_DATABASE_PATH.write_text(json.dumps(database, indent=2), encoding="utf-8")
        print("Deleted only explicitly eligible non-finalized records.")
    elif args.apply_expired or args.apply_orphans:
        print("No eligible records to delete.")
    else:
        print("Dry run only. Use --apply-expired to delete eligible temporary records.")


if __name__ == "__main__":
    main()
