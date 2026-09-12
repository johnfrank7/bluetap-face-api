# Face storage lifecycle

Before this change, `face-data/embeddings.json` had one `subjects[]` collection:

```json
{ "subjects": [{ "subject_id": "registration-or-user-id", "embedding": ["..."], "created_at": "..." }] }
```

Every entry was searched for duplicates, so an abandoned registration could act as a permanent match.

After this change, the file has two collections:

```json
{
  "subjects": [{
    "subject_id": "<trusted Firebase uid>",
    "embedding": ["..."],
    "finalized": true,
    "registration_session_id": "<session id>",
    "created_at": "..."
  }],
  "temporary_registrations": [{
    "registration_session_id": "<session id>",
    "embedding": ["..."],
    "created_at": "...",
    "expires_at": "..."
  }]
}
```

`/check-duplicate` searches only records with `finalized: true`. Temporary records expire automatically (one hour by default; configure `REGISTRATION_FACE_TTL_SECONDS`) and may be deleted by the trusted server through `/discard-registration-face`.

The authenticated flow is `/store-registration-face` followed by `/enroll-face` with a trusted Firebase `uid` and `registration_session_id`. `/enroll-face` no longer accepts a browser-provided subject ID or image. Keep `DEEPFACE_API_KEY` server-side.

Existing records without `finalized: true` are neither automatically promoted nor deleted. The service fails closed while any such record exists, so reconcile them with trusted completed-account data before deployment. `python cleanup_registration_faces.py` is a dry run; `--apply-expired` deletes only expired temporary records. To remove an old orphaned legacy record, first create a reviewed JSON array of its subject IDs, then use `--orphan-subjects reviewed.json --apply-orphans`. Neither mode can delete a record with `finalized: true`.
