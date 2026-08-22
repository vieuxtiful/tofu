"""Train the Phase 1 typeface-invariant proof encoder on synthetic text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read_lines(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


## Attempts per instance for the distributional objective: the observed
## views of one piece of ink. Mirrors what `proof_runtime._views` scores at
## inference -- the clean read plus a blur and a morphological close --
## because supervising a set of attempts the runtime does not produce would
## train a consistency the deployed system never has to satisfy.
DISTRIBUTIONAL_PROCESSES = ("blur", "resample", "abrasion")

## Candidates per instance. The truth plus its hard negatives, capped:
## `proof.hard_negatives` returns silhouette confusions, of which a long
## string has many, and an unbounded pool would make one long region cost
## what fifty short ones do.
DISTRIBUTIONAL_POOL = 8


def _training_pool(text, vocabulary, pool_size):
    """The candidate slate for one training instance.

    Built the way `proof_runtime.candidate_pool` builds one at inference,
    and for the same reasons: measured hard negatives first, then other
    strings the system could have proposed. Note what inference actually
    does -- it calls `proof.hard_negatives(inst.text)` on the OBSERVED
    READ, not on any truth it does not have. Here the rendered ink is the
    truth, so feeding the truth to the same generator is that generator
    fed the same kind of input, not a peek at the answer.

    On CJK the silhouette-confusion table is empty -- its pairs are Latin
    -- so the fill is not a garnish, it is the whole pool. Without it every
    zh-Hant instance would have a pool of one and be skipped, which is how
    an objective can look implemented and never run.

    THE FILL MUST NOT RANK BY DISTANCE TO THE TRUTH. Choosing the nearest
    strings would make a harder and more useful pool, and would also make
    pool membership a function of the answer in a way inference never is.
    It is a deterministic pseudo-random slice instead, and the result is
    SHUFFLED -- a pool whose truth always sits first teaches position, not
    identity.
    """
    import random

    from tofu.layers import proof, proof_encoder

    pool = [text]
    for item in proof.hard_negatives(text, limit=pool_size - 1):
        if item["text"] not in pool:
            pool.append(item["text"])
    others = [other for other in vocabulary if other != text]
    ## Seeded from the string itself, so the same instance draws the same
    ## slate on every run and two arms can be compared on identical pools.
    chooser = random.Random(proof_encoder.stable_bucket(text, buckets=(1 << 31) - 1))
    chooser.shuffle(others)
    for other in others:
        if len(pool) >= pool_size:
            break
        if other not in pool:
            pool.append(other)
    chooser.shuffle(pool)
    return pool


def _instances(texts, fonts, processes, pool_size=DISTRIBUTIONAL_POOL):
    """Attempt-level training instances for the distributional objective.

    One instance is a region-shaped thing: a candidate pool, a truth, and
    several ATTEMPTS at reading the same ink. That shape is what makes the
    multi-view machinery trainable -- the incumbent loss sees a flat bag of
    images and cannot express "these three are the same observation".

    Synthetic ink, so every instance is `present`: the mark is rendered
    intact and then degraded by a named process, which is evidence that
    survives by construction. The survival gate is therefore exercised by
    the test suite rather than by this data, and a real-ink corpus is what
    puts weak and absent instances in front of it.
    """
    from tofu.layers import proof, proof_encoder

    rows = []
    for text in texts:
        pool = _training_pool(text, texts, pool_size)
        if len(pool) < 2:
            continue
        for font in fonts:
            clean = proof.render(text, font)
            if clean is None:
                continue
            attempts = [proof_encoder.prepare(clean)]
            for process in processes:
                damaged = proof.degrade(clean, process, proof.MODERATE, seed=(text, font, process))
                if damaged is not None:
                    attempts.append(proof_encoder.prepare(damaged))
            candidates = []
            renderable = []
            for candidate in pool:
                image = proof.render(candidate, font)
                if image is None:
                    continue
                renderable.append(candidate)
                candidates.append(proof_encoder.prepare(image))
            ## A pool that lost its truth to a missing glyph is not a
            ## harder instance, it is an unanswerable one: the target would
            ## put its mass on whichever wrong string happened to render.
            if text not in renderable or len(renderable) < 2:
                continue
            rows.append({
                "text": text, "pool": renderable, "font": font,
                "attempts": attempts, "candidates": candidates,
                ## One channel per instance. Consistency is required across
                ## the attempts of one measurement and nowhere else -- no
                ## cross-channel tie is certified; see
                ## docs/channel-envelope-report.md.
                "channels": ["synthetic_render"] * len(attempts),
                "survival": "present",
            })
    return rows


def _distributional_epoch(model, rows, optimizer, batch_size, order):
    """One pass over the instances, in minibatches.

    Minibatched because the incumbent loop stacks every image into a single
    tensor and steps once per epoch -- workable for a flat sample bag, not
    for attempt-level data, where the image count is multiplied by attempts
    plus pool size per instance.

    Every image in a batch goes through the encoder in ONE forward pass and
    is sliced afterwards, so batching buys the throughput it is meant to;
    encoding instance by instance inside the loop would keep the memory
    profile and lose the speed.
    """
    import numpy as np
    import torch

    from tofu.layers import proof_objective

    totals = {"loss": 0.0, "cross_entropy": 0.0, "consistency": 0.0, "instances": 0}
    for start in range(0, len(order), batch_size):
        batch = [rows[index] for index in order[start : start + batch_size]]
        images, spans = [], []
        for row in batch:
            attempt_start = len(images)
            images.extend(row["attempts"])
            candidate_start = len(images)
            images.extend(row["candidates"])
            spans.append((attempt_start, candidate_start, len(images)))
        tensor = torch.from_numpy(np.stack(images)).unsqueeze(1)
        vectors = model(tensor)

        optimizer.zero_grad()
        loss = vectors.sum() * 0.0
        for row, (attempt_start, candidate_start, end) in zip(batch, spans, strict=True):
            observed = vectors[attempt_start:candidate_start]
            candidates = vectors[candidate_start:end]
            result = proof_objective.instance_objective(
                observed @ candidates.T, row["pool"], row["text"], row["survival"],
                channels=row["channels"],
            )
            loss = loss + result["loss"]
            totals["cross_entropy"] += float(result["cross_entropy"].detach())
            totals["consistency"] += float(result["consistency"].detach())
            totals["instances"] += 1
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.detach())
    return totals


def _samples(texts, fonts, processes, vertical_augmentation=False):
    import numpy as np

    from tofu.layers import proof, proof_encoder

    rows = []
    for label, text in enumerate(texts):
        for font in fonts:
            clean = proof.render(text, font)
            if clean is None:
                continue
            rows.append((proof_encoder.prepare(clean), label, text, font, "clean"))
            if vertical_augmentation:
                rows.append((proof_encoder.prepare(np.rot90(clean, -1)), label, text, font, "vertical"))
            for process in processes:
                damaged = proof.degrade(clean, process, proof.MODERATE, seed=(text, font, process))
                rows.append((proof_encoder.prepare(damaged), label, text, font, process))
                if vertical_augmentation:
                    rows.append((
                        proof_encoder.prepare(np.rot90(damaged, -1)),
                        label,
                        text,
                        font,
                        f"vertical-{process}",
                    ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--train-font", action="append", required=True)
    parser.add_argument("--holdout-font", action="append", required=True)
    parser.add_argument("--ground-truth", action="append", default=[])
    parser.add_argument("--ground-truth-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vertical-augmentation", action="store_true")
    parser.add_argument("--length-weight", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=12)
    ## The incumbent stays the default. A new objective may sit alongside
    ## the shipped one and be measured against it; it may not replace it
    ## before a paired gate says it should.
    parser.add_argument(
        "--objective", choices=("contrastive", "distributional"),
        default="contrastive",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import numpy as np
    import torch

    from tofu.layers import proof_encoder

    torch.manual_seed(args.seed)
    texts = _read_lines(args.texts)
    ground_truth = {text.strip() for text in args.ground_truth if text.strip()}
    if args.ground_truth_file:
        ground_truth.update(_read_lines(args.ground_truth_file))
    reserved = ground_truth | {
        char for text in ground_truth for char in text if not char.isspace()
    }
    available = [text for text in texts if text not in reserved]
    train_texts = [text for text in available if proof_encoder.split_for(text) == "train"]
    holdout_texts = sorted(reserved | {
        text for text in available if proof_encoder.split_for(text) == "holdout"
    })
    train_rows = _samples(
        train_texts,
        args.train_font,
        ("blur", "resample", "abrasion"),
        args.vertical_augmentation,
    )
    holdout_rows = _samples(
        holdout_texts, args.holdout_font, (), args.vertical_augmentation
    )
    if not train_rows or not holdout_rows:
        raise SystemExit("both train and held-out samples are required")

    multitask = args.length_weight > 0
    model = (
        proof_encoder.build_multitask_model(max_length=args.max_length)
        if multitask else proof_encoder.build_model()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    images = torch.from_numpy(np.stack([row[0] for row in train_rows])).unsqueeze(1)
    labels = torch.tensor([row[1] for row in train_rows], dtype=torch.long)
    lengths = torch.tensor(
        [min(len(row[2]), args.max_length) - 1 for row in train_rows],
        dtype=torch.long,
    )
    model.train()
    losses = []
    objective_report: dict = {}
    if args.objective == "distributional":
        import random

        from tofu.layers import proof_objective

        instances = _instances(
            train_texts, args.train_font, DISTRIBUTIONAL_PROCESSES,
        )
        if not instances:
            raise SystemExit("no distributional instances could be built")
        shuffler = random.Random(args.seed)
        order = list(range(len(instances)))
        for _ in range(args.epochs):
            shuffler.shuffle(order)
            totals = _distributional_epoch(
                model, instances, optimizer, args.batch_size, order,
            )
            losses.append(totals["loss"])
        objective_report = {
            "instances": len(instances),
            "attempts_per_instance": 1 + len(DISTRIBUTIONAL_PROCESSES),
            "pool_size_mean": round(
                sum(len(row["pool"]) for row in instances) / len(instances), 2
            ),
            "batch_size": args.batch_size,
            "temperature": proof_objective.TEMPERATURE,
            "sigma": proof_objective.SIGMA,
            "mu_consistency": proof_objective.MU_CONSISTENCY,
            "final_cross_entropy": totals["cross_entropy"] / max(1, totals["instances"]),
            "final_consistency": totals["consistency"] / max(1, totals["instances"]),
            ## Synthetic ink survives by construction, so the gate never
            ## fires here. Said in the report rather than left to be
            ## inferred from a number that looks like full coverage.
            "survival_states": {"present": totals["instances"]},
        }
    else:
        for _ in range(args.epochs):
            optimizer.zero_grad()
            if multitask:
                embeddings, length_logits = model.forward_with_length(images)
                contrastive = proof_encoder.supervised_contrastive_loss(embeddings, labels)
                length_loss = torch.nn.functional.cross_entropy(length_logits, lengths)
                loss = contrastive + args.length_weight * length_loss
            else:
                loss = proof_encoder.supervised_contrastive_loss(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

    holdout_images = torch.from_numpy(np.stack([row[0] for row in holdout_rows])).unsqueeze(1)
    with torch.no_grad():
        if multitask:
            vectors, holdout_length_logits = model.forward_with_length(holdout_images)
        else:
            vectors = model(holdout_images)
        similarities = vectors @ vectors.T
    correct = 0
    for index, row in enumerate(holdout_rows):
        scores = similarities[index].clone()
        scores[index] = float("-inf")
        prediction = holdout_rows[int(scores.argmax())][2]
        correct += prediction == row[2]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    holdout_lengths = [min(len(row[2]), args.max_length) for row in holdout_rows]
    length_correct = None
    if multitask:
        predictions = holdout_length_logits.argmax(dim=1).add(1).tolist()
        length_correct = sum(
            predicted == truth for predicted, truth in zip(predictions, holdout_lengths, strict=False)
        )
    torch.save({
        "schema": 2 if multitask else 1,
        "architecture": "multitask" if multitask else "embedding",
        "model": model.state_dict(),
        "embedding_dim": proof_encoder.EMBEDDING_DIM,
        "max_length": args.max_length if multitask else None,
    }, args.out)
    report = {
        "schema": 2 if multitask else 1,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_strings": len(train_texts),
        "holdout_strings": len(holdout_texts),
        "reserved_ground_truth": sorted(reserved),
        "train_fonts": args.train_font,
        "holdout_fonts": args.holdout_font,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "holdout_cross_face_top1": correct / len(holdout_rows),
        "vertical_augmentation": args.vertical_augmentation,
        "length_weight": args.length_weight,
        "holdout_length_top1": (
            length_correct / len(holdout_rows) if length_correct is not None else None
        ),
        "objective": args.objective,
        "checkpoint": str(args.out),
    }
    if objective_report:
        report["distributional"] = objective_report
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
