# Native IME providers

ToFU's browser editor does not replace, inspect, style, or switch the host
operating system IME. The editor's `lang` attribute is a language hint only;
Windows remains authoritative for composition and candidate selection.

Rime and Mozc belong to a separate native feature package:

| Provider | Scope | Engine | Browser bundle |
| --- | --- | --- | --- |
| Rime | Chinese schemas and dictionaries | `librime` | Never |
| Mozc | Japanese kana/kanji conversion | Mozc converter | Never |

The native package must expose a narrow local interface rather than injecting
keystroke handling into React:

```text
startSession(language, projectId)
updatePreedit(sessionId, reading)
listCandidates(sessionId, page, pageSize)
commitCandidate(sessionId, candidateId)
endSession(sessionId)
```

Candidate records include `text`, `reading`, `provider`, `rank`, and page
metadata. ToFU may re-rank valid provider candidates with asset/project Ground
Truth and accepted project history, but it must not label a user commit as an
OCR correction.

Packaging requirements:

- Rime/Mozc binaries and dictionary notices are installed as optional native
  components and are not downloaded during ordinary browser startup.
- Provider availability is discovered through the local backend.
- Browser mode remains the fallback when no native provider is installed.
- The native host is responsible for Windows TSF/input-language integration;
  the web application never claims that a `lang` attribute changed the active
  Windows keyboard.

