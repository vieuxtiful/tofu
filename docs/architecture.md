# ToFU Architecture

This document provides a visual overview of the ToFU system architecture,
complementing the detailed [cross-layer architecture](cross-layer-architecture.md)
and the [API reference](api.md).

## System Overview

ToFU is a three-tier system: a React/Vite frontend, a FastAPI server, and a
Python pipeline library (`tofu-l10n`). The frontend communicates with the
server via REST; the server orchestrates the pipeline layers and manages
SQLite-backed project state.

```mermaid
graph TB
    subgraph Frontend["Frontend (React + Vite)"]
        UI["App.tsx<br/>Main workspace"]
        VW["VideoWorkspace.tsx<br/>Video editor"]
        RT["RegionTable.tsx<br/>Region grid"]
        SSP["SemanticSubstitutionPanel.tsx<br/>Basil units"]
        Hooks["useManifest / useGlossary<br/>useSemanticUnits"]
    end

    subgraph Server["FastAPI Server (server/main.py)"]
        API["81 REST endpoints"]
        DB["SQLite (server/db.py)"]
        Worker["Video worker thread"]
        FS["File storage<br/>(uploads/, outputs/)"]
    end

    subgraph Pipeline["tofu-l10n Library (src/tofu/)"]
        Core["core/types.py<br/>TextManifest, BBox, StyleProfil"]
        Layers["layers/<br/>7 pipeline stages"]
        Video["video/<br/>Braise + Compositor"]
        Utils["utils/<br/>XLIFF, VTM, interchange"]
    end

    UI -->|HTTP/REST| API
    VW -->|HTTP/REST| API
    API --> DB
    API --> FS
    API --> Layers
    API --> Video
    Worker --> Video
    Worker --> DB
    Layers --> Core
    Video --> Core
```

## Image Pipeline (7 Layers)

The image pipeline processes a single image through a fixed stage order.
Each layer receives and enriches a `TextManifest` — the single data structure
that flows through the entire pipeline.

```mermaid
flowchart LR
    Input["Input Image<br/>(PNG/JPG)"]
    Tofu["ToFU<br/>pre-flight validation"]
    Scene1["Scene (pre-pass)<br/>surface detection"]
    Cicerone["Cicerone<br/>text detection + OCR"]
    Tofu2["ToFU (re-validate)<br/>glyph coverage check"]
    Scene2["Scene (enrich)<br/>per-instance style"]
    Cleanse["Cleanse<br/>text erasure + inpainting"]
    Scribe["Scribe<br/>target text rendering"]
    Verify["Verify<br/>quality scoring"]
    Memory["Memory<br/>translation memory store"]

    Input --> Tofu
    Tofu --> Scene1
    Scene1 --> Cicerone
    Cicerone --> Tofu2
    Tofu2 --> Scene2
    Scene2 --> Cleanse
    Cleanse --> Scribe
    Scribe --> Verify
    Verify --> Memory

    subgraph Manifest["TextManifest (flows through all stages)"]
        direction LR
        M1["instances[]<br/>text, bbox, confidence"]
        M2["style_profile<br/>font, color, weight"]
        M3["target_text<br/>translation"]
        M4["qa_score<br/>verification"]
    end
```

### Auxiliary Modules

Three auxiliary modules run alongside the main pipeline:

```mermaid
graph LR
    subgraph Auxiliary["Auxiliary Modules"]
        Savor["Savor<br/>glyph correction"]
        Wasabi["Wasabi<br/>CJK normalization"]
        Menu["Menu<br/>gazetteer recovery"]
    end

    Cicerone["Cicerone output"] --> Savor
    Savor --> Wasabi
    Wasabi --> Menu
    Menu --> Cleanse
```

## Video Pipeline

The video pipeline extends the image workflow with temporal tracking and
frame-by-frame compositing.

```mermaid
flowchart LR
    subgraph Ingest["Ingest"]
        Probe["Probe video<br/>(FFmpeg)"]
        Proxy["Create proxy<br/>(720p MP4)"]
        Chunk["Chunk frame range<br/>(240 frames/chunk)"]
    end

    subgraph Analysis["Analysis (Braise)"]
        Decode["Decode frame<br/>(cv2.VideoCapture)"]
        Detect["Scene cut?"]
        OCR["Run OCR<br/>(adaptive trigger)"]
        Flow["Optical flow<br/>(Lucas-Kanade)"]
        Track["Temporal tracking<br/>(association + consensus)"]
        Checkpoint["Checkpoint<br/>(every chunk)"]
    end

    subgraph Review["Review"]
        Timeline["Timeline view<br/>(tracks, observations)"]
        Edit["Track editing<br/>(text, style, keyframes)"]
        Preview["Preview render<br/>(frame range)"]
    end

    subgraph Export["Export (Compositor)"]
        Erase["Erase source text<br/>(temporal reconstruction)"]
        Render["Render target text<br/>(scribe)"]
        Composite["Composite frames"]
        Encode["Encode MP4<br/>(cv2.VideoWriter)"]
    end

    Probe --> Proxy --> Chunk
    Chunk --> Decode
    Decode --> Detect
    Detect -->|cut| OCR
    Detect -->|no cut| Flow
    OCR --> Track
    Flow --> Track
    Track --> Checkpoint
    Checkpoint -->|next chunk| Decode
    Checkpoint -->|done| Timeline
    Timeline --> Edit
    Edit --> Preview
    Edit --> Export
    Preview --> Export
    Erase --> Render --> Composite --> Encode
```

### Temporal Tracking Edge Cases

```mermaid
stateDiagram-v2
    [*] --> Pending: New detection
    Pending --> Active: K consecutive observations
    Pending --> [*]: Missed before confirm (flicker)

    Active --> Active: Matched (IoU + text similarity)
    Active --> Dormant: Missed K frames
    Active --> [*]: Text change split

    Dormant --> Active: Reappeared within gap window
    Dormant --> [*]: Gap exceeded MAX_GAP_FRAMES
```

## Three-Venv Architecture

ToFU uses three separate Python virtual environments to isolate incompatible
dependencies (EasyOCR needs torch, PaddleOCR needs its own protobuf, the
server needs FastAPI).

```mermaid
graph TB
    subgraph Main[".venv (main)"]
        Torch["PyTorch + EasyOCR"]
        FastAPI["FastAPI + Uvicorn"]
        CV2["OpenCV + scikit-image"]
        Core["tofu-l10n (editable)"]
    end

    subgraph Paddle[".venv-paddle"]
        PaddleOCR["PaddleOCR"]
        PaddleProto["protobuf 3.x"]
    end

    subgraph Inpaint[".venv-inpaint"]
        BrushNet["BrushNet + diffusers"]
        Lama["LaMa"]
    end

    Server["server/main.py"] -->|subprocess| Paddle
    Server -->|HTTP| Inpaint
    Server --> Main
```

## Data Flow: Frontend ↔ Server ↔ Pipeline

```mermaid
sequenceDiagram
    participant F as Frontend
    participant S as FastAPI Server
    participant P as Pipeline (tofu-l10n)
    participant D as SQLite DB

    F->>S: POST /api/assets (upload image)
    S->>D: create asset record
    S->>F: { asset_id }

    F->>S: POST /api/detect (run OCR)
    S->>P: cicerone.detect(image)
    P-->>S: TextManifest with instances
    S->>D: store manifest
    S-->>F: manifest JSON

    F->>S: PUT /api/manifest (update target text)
    S->>D: update regions
    S-->>F: updated manifest

    F->>S: POST /api/render (render localized image)
    S->>P: cleanse.erase() + scribe.render()
    P-->>S: rendered image
    S->>D: save output
    S-->>F: { output_url }

    F->>S: POST /api/export (XLIFF/VTM)
    S->>P: interchange.export_xliff()
    P-->>S: XLIFF XML
    S-->>F: file download
```

## Project Layout

```mermaid
graph LR
    Root["project-tofu/"]

    Root --> Src["src/tofu/"]
    Src --> Core["core/<br/>types, pipeline"]
    Src --> Layers["layers/<br/>7 pipeline stages"]
    Src --> Video["video/<br/>Braise, Compositor"]
    Src --> Utils["utils/<br/>XLIFF, VTM"]

    Root --> Server["server/"]
    Server --> Main["main.py<br/>FastAPI app"]
    Server --> DB["db.py<br/>SQLite layer"]

    Root --> Frontend["frontend/"]
    Frontend --> App["src/App.tsx"]
    Frontend --> Components["src/*.tsx<br/>components"]
    Frontend --> Hooks["src/use*.ts<br/>hooks"]

    Root --> Tests["tests/"]
    Tests --> Unit["test_*.py<br/>unit tests"]
    Tests --> Regression["regression/<br/>baseline tests"]
    Tests --> E2E["test_server_e2e.py<br/>E2E tests"]

    Root --> Scripts["scripts/"]
    Scripts --> Eval["eval_*.py<br/>evaluation harnesses"]
    Scripts --> Gen["generate_api_docs.py<br/>doc generation"]
    Scripts --> Fixtures["video_fixtures.py<br/>test clips"]
```

## Key Design Principles

1. **Single data structure**: A `TextManifest` flows through every pipeline
   layer. No layer reaches into another's internals — they exchange the
   manifest and nothing else.

2. **Storage-agnostic layers**: Pipeline layers are pure functions that take
   and return manifests. The server owns all persistence (SQLite, file I/O).

3. **Three-venv isolation**: Incompatible dependencies (torch vs PaddleOCR vs
   diffusers) are isolated in separate venvs, communicating via subprocess/HTTP.

4. **Resumable video jobs**: The Braise tracker captures its state at chunk
   boundaries as a `TrackerCheckpoint`, enabling exact resume after interruption.

5. **Preview/export parity**: Video preview frames are byte-identical to export
   frames on their shared range — a preview is a window onto the export, not a
   different render.
