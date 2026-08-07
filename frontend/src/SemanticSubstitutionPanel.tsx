import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Check, ChevronDown, Loader2, Plus, RotateCcw, ScanText, ShieldAlert, Trash2, X } from "lucide-react";
import { FcCollapse } from "react-icons/fc";
import { MdTipsAndUpdates } from "react-icons/md";
import { CgArrangeBack } from "react-icons/cg";
import { TbLeafFilled } from "react-icons/tb";
import { RiLightbulbAiLine, RiLightbulbAiFill } from "react-icons/ri";
import type { GlossaryStatus, InstText, SemanticSubstitutionPlan, SemanticTextUnit } from "./api";
import type { Theme } from "./theme";
import GlossaryPanel from "./GlossaryPanel";
import { regionColorMap } from "./regionPalette";
import { guardTargetText, type AttestedSet, type GuardVerdict } from "./targetGuard";
import AnimatedCaretTextarea from "./AnimatedCaretTextarea";
import basilHeaderUrl from "../../images/basil-header-light.png";
import basilHeaderDarkUrl from "../../images/basil-header-dark.png";

const PLACEHOLDER_TEXT = "Enter target phrase";

type Props = {
  units: SemanticTextUnit[];
  drafts: Record<string, string>;
  plans: Record<string, SemanticSubstitutionPlan>;
  busyId: string | null;
  onDraftChange: (unitId: string, text: string) => void;
  onPlan: (unitId: string, apply: boolean) => void;
  onRepair: (unitId: string, accepted: boolean) => void;
  onModifyMembers: (unitId: string, options: { addRegionId?: string; removeRegionId?: string }) => void;
  onCreateUnit: () => void;
  onDeleteUnit: (unitId: string) => void;
  theme: Theme;
  projectId: string | null;
  targLang: string;
  regionsById: Record<string, InstText>;
  attested: AttestedSet;
  glossaryStatus: GlossaryStatus | null;
  onGlossaryUpload: (file: File, scope: "global" | "project", mode: "auxiliary" | "merge" | "replace") => void;
  onGlossaryDelete: (scope: "global" | "project") => void;
  glossaryUploading: boolean;
  glossaryUploadStep: "marinate" | "ferment" | "set" | "done" | null;
  glossaryUploadError: string | null;
  sourceSuggestions?: string[];
};

/** Languages written without spaces between words. Their target fields
 * cannot be token-coloured as the user types — there is no delimiter to
 * key off — so they get ordered slots instead of a free-text box. */
const UNSPACED_TARGETS = /^(ja|ko|th|km|lo|my|zh)(-|$)/i;

const WORD_RE = /[\p{L}\p{M}\p{N}][\p{L}\p{M}\p{N}‐-―'’-]*/gu;

function normToken(word: string): string {
  return word.toLowerCase().replace(/[‐-―'’\-\s]/g, "");
}

/**
 * Makes the semantic and spatial orders visible at the same time.  The
 * explicit Apply button is a safety boundary: entering a natural target
 * phrase never silently replaces the per-region target fields.
 */
export default function SemanticSubstitutionPanel({
  units, drafts, plans, busyId, onDraftChange, onPlan, onRepair, onModifyMembers, onCreateUnit, onDeleteUnit, theme, projectId,
  targLang, regionsById, attested, glossaryStatus,
  onGlossaryUpload, onGlossaryDelete, glossaryUploading, glossaryUploadStep, glossaryUploadError,
  sourceSuggestions = [],
}: Props) {
  // Show multi-region units AND user-created plates (which may start empty
  // or with a single region).  The panel always renders so the "add plate"
  // button is available even when no auto-detected multi-region units exist.
  const multiRegion = units.filter(
    (unit) => unit.region_ids.length > 1 || unit.analysis_provider === "user_created",
  );
  const [open, setOpen] = useState(false);
  const [slotOrders, setSlotOrders] = useState<Record<string, string[]>>({});
  // Plates the user has dismissed this session.  Dismissal is local UI state
  // only — the units stay in the manifest and each leaves behind a "show X"
  // restore button in its place.
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  if (multiRegion.length === 0 && !units.some((u) => u.analysis_provider === "user_created")) return null;
  return (
    <section className="relative isolate rounded-xl border border-cyan-200/80 bg-cyan-50/55 p-3 dark:border-cyan-900/80 dark:bg-cyan-950/20">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit]">
        <img
          alt=""
          src={theme === "dark" ? basilHeaderDarkUrl : basilHeaderUrl}
          className={`absolute -top-8 inset-x-0 w-full object-cover object-top transition-opacity duration-300 [mask-image:linear-gradient(to_bottom,black_0%,black_58%,transparent_96%,transparent_100%)] ${open ? "opacity-100" : "opacity-55"}`}
        />
      </div>
      <div className="relative z-10">
      <div className="flex items-center gap-2 pr-7 text-sm font-semibold text-cyan-950 dark:text-cyan-100">
        <TbLeafFilled size={15} className="text-emerald-600" />
        <span>Basil</span>
      </div>
      <button type="button" onClick={() => setOpen((value) => !value)} className="absolute right-0 top-0 rounded-sm p-1 text-zinc-500 transition hover:bg-zinc-200 dark:hover:bg-zinc-800" title={open ? "collapse Basil" : "expand Basil"} aria-label={open ? "Collapse Basil" : "Expand Basil"}>
        <FcCollapse size={12} style={{ transform: open ? "none" : "rotate(180deg)", transition: "transform 0.2s" }} />
      </button>
      <p className="subtext mb-1 flex items-center gap-1 text-xs text-zinc-500">
        <MdTipsAndUpdates size={14} className="shrink-0 text-[#2d8cf0]" />
        <span>Set target arrangement and review before plating.</span>
      </p>
      <div className={`style-panel-morph${open ? " expanded" : ""}`}>
      <div>
      <GlossaryPanel theme={theme} projectId={projectId} status={glossaryStatus} onUpload={onGlossaryUpload} onDelete={onGlossaryDelete} uploading={glossaryUploading} uploadStep={glossaryUploadStep} uploadError={glossaryUploadError} />
      <div className="mt-3 space-y-3">
        {multiRegion.map((unit) =>
          dismissedIds.has(unit.id) ? (
            <div key={unit.id} className="basil-dismissed-row">
              <button
                type="button"
                onClick={() =>
                  setDismissedIds((prev) => {
                    const next = new Set(prev);
                    next.delete(unit.id);
                    return next;
                  })
                }
                className="basil-show-plate-btn"
                title={`show "${unit.source_text}"`}
                aria-label={`show plate ${unit.source_text}`}
              >
                <RotateCcw size={12} />
                <span className="truncate">{unit.source_text}</span>
              </button>
              <button
                type="button"
                onClick={() => onDeleteUnit(unit.id)}
                className="basil-delete-plate-btn"
                title="delete plate"
                aria-label="Delete plate"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ) : (
            <BasilUnitCard
              key={unit.id}
              unit={unit}
              plan={plans[unit.id]}
              value={drafts[unit.id] ?? unit.substitution?.target_text ?? unit.suggestion?.target_text ?? ""}
              prefilled={drafts[unit.id] === undefined && !unit.substitution?.target_text && !!unit.suggestion?.target_text}
              loading={busyId === unit.id}
              theme={theme}
              targLang={targLang}
              regionsById={regionsById}
              attested={attested}
              sourceSuggestions={sourceSuggestions}
              slotOrder={slotOrders[unit.id]}
              onSlotOrder={(order) => setSlotOrders((previous) => ({ ...previous, [unit.id]: order }))}
              onDraftChange={onDraftChange}
              onPlan={onPlan}
              onRepair={onRepair}
              onModifyMembers={onModifyMembers}
              onDismiss={() =>
                setDismissedIds((prev) => {
                  const next = new Set(prev);
                  next.add(unit.id);
                  return next;
                })
              }
            />
          ),
        )}
        <button
          type="button"
          onClick={onCreateUnit}
          className="basil-add-plate-btn"
          title="add plate"
          aria-label="Add plate"
        >
          <Plus size={16} />
        </button>
      </div>
      </div>
      </div>
      </div>
    </section>
  );
}

type CardProps = {
  unit: SemanticTextUnit;
  plan: SemanticSubstitutionPlan | undefined;
  value: string;
  prefilled: boolean;
  loading: boolean;
  theme: Theme;
  targLang: string;
  regionsById: Record<string, InstText>;
  attested: AttestedSet;
  sourceSuggestions?: string[];
  slotOrder: string[] | undefined;
  onSlotOrder: (order: string[]) => void;
  onDraftChange: (unitId: string, text: string) => void;
  onPlan: (unitId: string, apply: boolean) => void;
  onRepair: (unitId: string, accepted: boolean) => void;
  onModifyMembers: (unitId: string, options: { addRegionId?: string; removeRegionId?: string }) => void;
  onDismiss: () => void;
};

function BasilUnitCard({
  unit, plan, value, prefilled, loading, theme, targLang, regionsById, attested,
  sourceSuggestions = [],
  slotOrder, onSlotOrder, onDraftChange, onPlan, onRepair, onModifyMembers, onDismiss,
}: CardProps) {
  const colors = useMemo(() => regionColorMap(unit.region_ids), [unit.region_ids]);
  const verdict = unit.pairing?.verdict ?? "unknown";
  // A pairing that cannot reorder still shows its entity and any proposed
  // repair — only the pointless decision is withdrawn.
  const platable = verdict !== "unnecessary";
  const unspacedTarget = UNSPACED_TARGETS.test(targLang);
  const guard: GuardVerdict | null = platable && !unspacedTarget
    ? guardTargetText(value, targLang, attested)
    : null;
  const [expanded, setExpanded] = useState(true);
  const [showAddRegion, setShowAddRegion] = useState(false);
  const addRegionDropdownRef = useRef<HTMLDivElement>(null);

  // Regions available to add: all manifest regions not already in this unit.
  const availableRegionIds = useMemo(
    () => Object.keys(regionsById).filter((id) => !unit.region_ids.includes(id)),
    [regionsById, unit.region_ids],
  );

  // close add-region dropdown on outside click
  useEffect(() => {
    if (!showAddRegion) return;
    const onDown = (e: MouseEvent) => {
      if (addRegionDropdownRef.current && !addRegionDropdownRef.current.contains(e.target as Node)) {
        setShowAddRegion(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [showAddRegion]);

  // Which region's meaning each target token carries, best evidence first:
  // this session's plan, then a previously applied substitution (which is
  // what the pre-filled value came from), then the per-region translations
  // the user has entered by hand. An unmatched token stays neutral rather
  // than taking a colour it has not earned.
  const tokenRegion = useMemo(() => {
    const map = new Map<string, string>();
    const applied = unit.substitution?.applied ? unit.substitution.assignments ?? [] : [];
    const rows = plan?.assignments?.length
      ? plan.assignments.map((assignment) => ({ id: assignment.region_id, text: assignment.text }))
      : applied.length
        ? applied.map((assignment) => ({ id: assignment.region_id, text: assignment.text }))
        : unit.region_ids.map((regionId) => ({ id: regionId, text: regionsById[regionId]?.target_text ?? "" }));
    for (const row of rows) {
      for (const word of (row.text ?? "").match(WORD_RE) ?? []) {
        if (!map.has(normToken(word))) map.set(normToken(word), row.id);
      }
    }
    return map;
  }, [plan, unit.substitution, unit.region_ids, regionsById]);

  return (
    <div className="rounded-lg border border-cyan-200 bg-white/80 p-3 dark:border-cyan-900 dark:bg-zinc-950/45">
      <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 text-xs">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 text-zinc-500 transition hover:text-cyan-600 dark:hover:text-cyan-400"
            title={expanded ? "collapse" : "expand"}
            aria-label={expanded ? "collapse" : "expand"}
          >
            <ChevronDown size={13} className={`transition-transform duration-200 ${expanded ? "" : "-rotate-90"}`} />
            <span className="font-medium text-zinc-800 dark:text-zinc-100">{unit.source_text}</span>
          </button>
          <span className="text-zinc-400 dark:text-zinc-600 text-[10px]" aria-hidden="true">・</span>
          <span className="flex items-center gap-1.5 text-[10px] text-zinc-500 dark:text-zinc-400">
            <ScanText size={12} className="shrink-0 text-cyan-600 dark:text-cyan-400" />
            <span className="font-medium text-zinc-700 dark:text-zinc-300">{unit.entity_type.replace(/_/g, " ")}</span>
            <span className={unit.confidence >= 0.8 ? "text-emerald-600 dark:text-emerald-300" : unit.confidence >= 0.6 ? "text-amber-600 dark:text-amber-300" : "text-red-500 dark:text-red-400"}>
              {Math.round(unit.confidence * 100)}%
            </span>
          </span>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="basil-dismiss-btn"
          title="dismiss plate"
          aria-label="Dismiss plate"
        >
          <X size={12} />
        </button>
      </div>

      <div className={`style-panel-morph${expanded ? " expanded" : ""}`}>
      {/* the halo: one coloured square per source region, in reading order. */}
      <div className="basil-strip" role="list" aria-label="Source regions in reading order">
        {unit.region_ids.map((regionId, idx) => (
          <span key={regionId} role="listitem" className="basil-strip-item">
            <span
              className="basil-chip"
              style={{ "--bh-ink": colors[regionId] } as React.CSSProperties}
              title={`region ${regionId}`}
            >
              <span className="basil-chip-id">{regionId}</span>
              <span className="basil-chip-text">{regionsById[regionId]?.text ?? ""}</span>
              <button
                type="button"
                className="basil-chip-close"
                title={`Remove ${regionId} from this unit`}
                aria-label={`Remove ${regionId} from this unit`}
                disabled={loading}
                onClick={() => onModifyMembers(unit.id, { removeRegionId: regionId })}
              >
                <X size={10} />
              </button>
            </span>
          </span>
        ))}
        {/* add-region bubble: dotted-border, opens an animated dropdown */}
        <span className="basil-chip-add" role="listitem">
          <div className="relative" ref={addRegionDropdownRef}>
            <button
              type="button"
              className="basil-chip-add-btn"
              title="Add a region to this unit"
              aria-label="Add a region to this unit"
              disabled={loading || availableRegionIds.length === 0}
              onClick={() => setShowAddRegion((v) => !v)}
            >
              <Plus size={11} />
            </button>
            <div
              className={`dropdown-morph absolute left-0 top-full z-200 mt-1 w-40 rounded-lg border border-zinc-300 bg-white p-1 dark:border-zinc-700 dark:bg-zinc-900${showAddRegion ? " expanded" : ""}`}
              style={showAddRegion ? { boxShadow: "1px 1px 0 var(--bc-shadow), 2px 2px 6px rgba(0,0,0,0.06)" } : undefined}
            >
              {availableRegionIds.length === 0 ? (
                <p className="px-2 py-1 text-[10px] text-zinc-400">No other regions available</p>
              ) : (
                availableRegionIds.map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => { setShowAddRegion(false); onModifyMembers(unit.id, { addRegionId: id }); }}
                    className="flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-[10px] transition hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  >
                    <span className="font-mono font-medium text-cyan-600 dark:text-cyan-400">{id}</span>
                    <span className="truncate text-zinc-600 dark:text-zinc-400">{regionsById[id]?.text ?? ""}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        </span>
      </div>

      {unit.ocr_repair && (
        <div className="basil-repair">
          <ScanText size={12} className="shrink-0 text-amber-600 dark:text-amber-300" />
          <span className="basil-repair-read">{unit.ocr_repair.read}</span>
          <ArrowRight size={11} className="shrink-0 opacity-60" />
          <span className="basil-repair-proposed">{unit.ocr_repair.proposed}</span>
          <span className="basil-repair-meta">
            gazetteer · {Math.round(unit.ocr_repair.similarity * 100)}% · {unit.ocr_repair.diffs.length} glyph
            {unit.ocr_repair.diffs.length === 1 ? "" : "s"}
          </span>
          {unit.ocr_repair.accepted ? (
            <button type="button" className="basil-repair-btn" disabled={loading} onClick={() => onRepair(unit.id, false)}>
              <RotateCcw size={11} /> revert
            </button>
          ) : (
            <>
              <button type="button" className="basil-repair-btn accept" disabled={loading} onClick={() => onRepair(unit.id, true)}>
                <Check size={11} /> accept
              </button>
              <button type="button" className="basil-repair-btn" disabled={loading} onClick={() => onRepair(unit.id, false)}>
                <X size={11} /> reject
              </button>
            </>
          )}
        </div>
      )}

      {!platable ? (
        <p className="basil-verdict">
          {unit.pairing?.reasons?.join("; ") || "this pairing preserves source order"} — no plating required.
        </p>
      ) : (
        <>
          {verdict === "unknown" && (
            <p className="basil-verdict">{unit.pairing?.reasons?.join("; ")}</p>
          )}
          {prefilled && unit.suggestion?.basis && (
            <p className="basil-suggestion">
              {theme === "dark"
                ? <RiLightbulbAiFill size={11} className="shrink-0" />
                : <RiLightbulbAiLine size={11} className="shrink-0" />}
              suggested from your per-region translations — {unit.suggestion.basis}
            </p>
          )}

          {unspacedTarget ? (
            <SlotEditor
              unit={unit}
              colors={colors}
              regionsById={regionsById}
              order={slotOrder ?? (unit.suggestion?.region_order?.length ? unit.suggestion.region_order : unit.region_ids)}
              onOrder={(order) => {
                onSlotOrder(order);
                onDraftChange(unit.id, order.map((regionId) => regionsById[regionId]?.target_text ?? "").join(""));
              }}
            />
          ) : (
            <AnimatedCaretTextarea
              value={value}
              onChange={(text) => onDraftChange(unit.id, text)}
              label={`Target phrase for ${unit.source_text}`}
              placeholder={PLACEHOLDER_TEXT}
              language={targLang}
              suggestions={/^(ja|zh|ko)(-|$)/i.test(targLang) ? sourceSuggestions : []}
              statusMessage={guard && !guard.ok && value.trim() ? guard.reason : null}
              rows={1}
              className="basil-plate-input"
            />
          )}

          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => onPlan(unit.id, false)}
              disabled={loading || !value.trim() || (guard !== null && !guard.ok)}
              title={guard && !guard.ok ? guard.reason : undefined}
              className="inline-flex items-center justify-center gap-1 rounded-md bg-cyan-700 px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {loading ? <Loader2 size={13} className="animate-spin" /> : <CgArrangeBack size={15} />}
              arrange
            </button>
            {guard && !guard.ok && value.trim() && (
              <span className="basil-guard-reason">{guard.reason}</span>
            )}
          </div>

          {plan && (
            <div className="mt-2 rounded-md bg-zinc-100/80 p-2 text-xs dark:bg-zinc-900/80">
              {plan.assignments.length > 0 ? (
                <>
                  <div className="flex flex-wrap gap-1.5 text-zinc-800 dark:text-zinc-100">
                    {plan.assignments.map((assignment) => (
                      <span
                        key={assignment.region_id}
                        className="basil-assign"
                        style={{ "--bh-ink": colors[assignment.region_id] } as React.CSSProperties}
                      >
                        <span className="basil-assign-id">
                          {assignment.region_id}
                          {assignment.anchor_id && assignment.anchor_id !== assignment.region_id ? ` → ${assignment.anchor_id}` : ""}
                        </span>{" "}
                        {assignment.text}
                      </span>
                    ))}
                  </div>
                  <p className="mt-1 text-zinc-500 dark:text-zinc-400">
                    blocks: <span className="font-mono">{plan.target_region_order.join(" → ")}</span>
                    <span className="mx-1">·</span>
                    cubes: <span className="font-mono">{plan.source_region_order.join(" → ")}</span>
                  </p>
                  <button
                    type="button"
                    onClick={() => onPlan(unit.id, true)}
                    disabled={loading || plan.review_required || (guard !== null && !guard.ok)}
                    title={guard && !guard.ok ? guard.reason : undefined}
                    className="mt-2 inline-flex items-center gap-1 rounded-md bg-emerald-700 px-2.5 py-1.5 text-xs font-medium text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    {loading ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
                    Plate cubes
                  </button>
                </>
              ) : (
                <div className="flex gap-1.5 text-amber-700 dark:text-amber-300">
                  <ShieldAlert size={14} className="mt-0.5 shrink-0" />
                  <span>{plan.warnings.join(" ") || "This phrase needs manual placement."}</span>
                </div>
              )}
            </div>
          )}
        </>
      )}
      </div>
    </div>
  );
}

/**
 * A transparent-text input layered over a mirror that renders the same
 * string as coloured spans.  contentEditable would be simpler to colour
 * and would break IME composition, selection and undo — disqualifying for
 * CJK entry — so the native input keeps all three and only its ink moves.
 */
function HaloField({
  value, guard, colors, tokenRegion, onChange, label,
}: {
  value: string;
  guard: GuardVerdict | null;
  colors: Record<string, string>;
  tokenRegion: Map<string, string>;
  onChange: (text: string) => void;
  label: string;
}) {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const [typedPlaceholder, setTypedPlaceholder] = useState(0);
  const offending = useMemo(() => new Set((guard?.offending ?? []).map(normToken)), [guard]);

  useEffect(() => {
    if (value) { setTypedPlaceholder(0); return; }
    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 1; i <= PLACEHOLDER_TEXT.length; i++) {
      timers.push(setTimeout(() => setTypedPlaceholder(i), i * 70));
    }
    return () => timers.forEach(clearTimeout);
  }, [value]);

  // Split on whitespace but KEEP the separators, so the mirror reproduces
  // the input string exactly and the two stay in glyph-for-glyph register.
  const pieces = useMemo(() => value.split(/(\s+)/), [value]);
  const lastWordIndex = useMemo(() => {
    for (let index = pieces.length - 1; index >= 0; index -= 1) {
      if (pieces[index] && !/^\s+$/.test(pieces[index])) return index;
    }
    return -1;
  }, [pieces]);
  // A token only takes its colour once the user has finished it with a
  // delimiter; the word still being typed stays neutral.
  const settled = value.length > 0 && /\s$/.test(value);

  return (
    <div className="basil-plate">
      <div className="basil-plate-mirror" aria-hidden="true" ref={mirrorRef}>
        {value === "" && (
          <span className="basil-placeholder-typewriter" style={{ color: "#a1a1aa" }}>
            {PLACEHOLDER_TEXT.slice(0, typedPlaceholder)}
            <span className="basil-placeholder-cursor" />
          </span>
        )}
        {pieces.map((piece, index) => {
          if (!piece) return null;
          if (/^\s+$/.test(piece)) return <span key={index}>{piece}</span>;
          const key = normToken(piece);
          const pending = index === lastWordIndex && !settled;
          if (offending.has(key)) {
            return <span key={index} className="basil-token reject">{piece}</span>;
          }
          const regionId = pending ? undefined : tokenRegion.get(key);
          if (!regionId) return <span key={index} className="basil-token">{piece}</span>;
          return (
            <span
              key={index}
              className="basil-token mapped"
              style={{ "--bh-ink": colors[regionId] ?? "currentColor" } as React.CSSProperties}
            >
              {piece}
            </span>
          );
        })}
      </div>
      <input
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onScroll={(event) => {
          if (mirrorRef.current) mirrorRef.current.scrollLeft = event.currentTarget.scrollLeft;
        }}
        className="basil-plate-input"
        spellCheck={false}
      />
    </div>
  );
}

/**
 * Ordered slots for targets written without spaces.  There is no delimiter
 * to colour tokens against in zh/ja/ko/th, so the arrangement is made by
 * moving the regions themselves rather than by typing a phrase.
 */
function SlotEditor({
  unit, colors, regionsById, order, onOrder,
}: {
  unit: SemanticTextUnit;
  colors: Record<string, string>;
  regionsById: Record<string, InstText>;
  order: string[];
  onOrder: (order: string[]) => void;
}) {
  const [dragging, setDragging] = useState<string | null>(null);
  const list = order.filter((regionId) => unit.region_ids.includes(regionId));
  const full = [...list, ...unit.region_ids.filter((regionId) => !list.includes(regionId))];

  const move = (from: string, to: string) => {
    if (from === to) return;
    const next = full.filter((regionId) => regionId !== from);
    next.splice(next.indexOf(to), 0, from);
    onOrder(next);
  };

  return (
    <div className="basil-slots" role="list" aria-label="Target arrangement">
      {full.map((regionId) => (
        <span
          key={regionId}
          role="listitem"
          draggable
          onDragStart={() => setDragging(regionId)}
          onDragEnd={() => setDragging(null)}
          onDragOver={(event) => event.preventDefault()}
          onDrop={() => { if (dragging) move(dragging, regionId); setDragging(null); }}
          className={`basil-slot${dragging === regionId ? " dragging" : ""}`}
          style={{ "--bh-ink": colors[regionId] } as React.CSSProperties}
        >
          <span className="basil-chip-id">{regionId}</span>
          <span className="basil-slot-text">
            {regionsById[regionId]?.target_text || <em className="opacity-50">untranslated</em>}
          </span>
        </span>
      ))}
    </div>
  );
}
