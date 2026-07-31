import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ChevronLeft, ChevronRight, Flag, Pause, Play, RotateCcw, X } from "lucide-react";
import { getVideoTimeline, RenderKeyframe, TextTrack, TrackObservation, VideoJob, VideoTimeline } from "./api";

interface Props {
  job: VideoJob; videoUrl: string;
  onKeyframe: (value: RenderKeyframe) => Promise<void>;
  onTrackUpdate?: (trackId: string, changes: Partial<Pick<TextTrack, "source_text"|"target_text"|"status">> & { expected_revision?: number }) => Promise<void>;
  onPreview?: (startFrame: number, endFrame: number) => Promise<void>; onExport?: () => Promise<void>;
  onCancel?: () => Promise<void>; onResume?: () => Promise<void>;
  onUpgrade?: () => Promise<void>;
}
type Scope = "frame" | "range" | "track";
const EMPTY: VideoTimeline = { tracks: [], observations: [], keyframes: [], issues: [] };
async function loadTimeline(jobId: string, start: number, end: number): Promise<VideoTimeline> {
  let cursor = 0, combined: VideoTimeline | null = null;
  do {
    const page = await getVideoTimeline(jobId, start, end, cursor);
    if (combined === null) combined = page;
    else combined = { tracks: combined.tracks, observations: combined.observations.concat(page.observations),
      keyframes: combined.keyframes, issues: combined.issues, truncated: page.truncated, next_cursor: page.next_cursor };
    cursor = page.next_cursor ?? 0;
    if (!page.truncated) break;
  } while (cursor);
  return combined ?? EMPTY;
}

function timecode(seconds: number) {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const hh = Math.floor(ms / 3600000), mm = Math.floor(ms / 60000) % 60, ss = Math.floor(ms / 1000) % 60;
  return `${String(hh).padStart(2,"0")}:${String(mm).padStart(2,"0")}:${String(ss).padStart(2,"0")}.${String(ms % 1000).padStart(3,"0")}`;
}
function nearestFrame(pts: number[], seconds: number, fps: number, max: number) {
  if (!pts.length) return Math.max(0, Math.min(max, Math.round(seconds * fps)));
  let lo = 0, hi = pts.length - 1;
  while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (pts[mid] < seconds) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(pts[lo - 1] - seconds) < Math.abs(pts[lo] - seconds)) lo--;
  return lo;
}

export default function VideoWorkspace({ job, videoUrl, onKeyframe, onTrackUpdate, onPreview, onExport, onCancel, onResume, onUpgrade }: Props) {
  const video = useRef<HTMLVideoElement>(null), stage = useRef<HTMLDivElement>(null), requestSeq = useRef(0);
  const cache = useRef(new Map<string, VideoTimeline>());
  const [frame, setFrame] = useState(0), [playing, setPlaying] = useState(false), [timeline, setTimeline] = useState<VideoTimeline>(EMPTY);
  const [selected, setSelected] = useState<string | null>(null), [scopePrompt, setScopePrompt] = useState(false), [scope, setScope] = useState<Scope>("track");
  const [rangeStart, setRangeStart] = useState<number | null>(null), [rangeEnd, setRangeEnd] = useState<number | null>(null);
  const [loading, setLoading] = useState(false), [error, setError] = useState<string | null>(null), [saving, setSaving] = useState(false);
  const [videoRect, setVideoRect] = useState({ left: 0, top: 0, width: 1, height: 1 });
  const [draftBox, setDraftBox] = useState<TrackObservation["bbox"] | null>(null), drag = useRef<{x:number;y:number;box:TrackObservation["bbox"]}|null>(null);
  const [sourceDraft, setSourceDraft] = useState(""), [targetDraft, setTargetDraft] = useState(""), [trackSaving, setTrackSaving] = useState(false);
  const [showLocalized, setShowLocalized] = useState(false);
  const [previewStart, setPreviewStart] = useState(0);
  const pts = job.manifest.frame_pts ?? [], max = Math.max(0, job.manifest.frame_count - 1), fps = job.manifest.fps || 30;
  const second = pts[frame] ?? frame / fps;
  const terminalFailure = job.status === "failed" || job.status === "cancelled";
  const ready = job.status === "ready" || job.status === "completed" || job.status === "rendering";
  const activeVideoUrl = showLocalized && job.preview_url ? job.preview_url : videoUrl;

  useEffect(() => { cache.current.clear(); setTimeline(EMPTY); }, [job.id]);
  useEffect(() => {
    if (!ready) return;
    const start = Math.max(0, Math.floor(frame / 300) * 300), end = Math.min(max, start + 599), key = `${start}:${end}`;
    const cached = cache.current.get(key); if (cached) { setTimeline(cached); return; }
    const seq = ++requestSeq.current; setLoading(true); setError(null);
    const timer = window.setTimeout(() => loadTimeline(job.id, start, end).then(value => {
      if (seq !== requestSeq.current) return; cache.current.set(key, value); setTimeline(value);
    }).catch(e => seq === requestSeq.current && setError(String(e))).finally(() => seq === requestSeq.current && setLoading(false)), 120);
    return () => window.clearTimeout(timer);
  }, [job.id, frame, max, ready]);

  useEffect(() => {
    const el = video.current; if (!el) return;
    const tick = () => setFrame(nearestFrame(pts, el.currentTime + (showLocalized ? (pts[previewStart] ?? previewStart/fps) : 0), fps, max));
    el.addEventListener("timeupdate", tick); return () => el.removeEventListener("timeupdate", tick);
  }, [fps, pts, max, showLocalized, previewStart]);

  useEffect(() => {
    const host = stage.current, el = video.current; if (!host || !el) return;
    const measure = () => {
      const w = host.clientWidth, h = host.clientHeight, sourceRatio = (el.videoWidth || job.manifest.width) / (el.videoHeight || job.manifest.height), hostRatio = w / h;
      const width = hostRatio > sourceRatio ? h * sourceRatio : w, height = hostRatio > sourceRatio ? h : w / sourceRatio;
      setVideoRect({ left: (w - width) / 2, top: (h - height) / 2, width, height });
    };
    const observer = new ResizeObserver(measure); observer.observe(host); el.addEventListener("loadedmetadata", measure); measure();
    return () => { observer.disconnect(); el.removeEventListener("loadedmetadata", measure); };
  }, [job.manifest.width, job.manifest.height]);

  const observations = useMemo(() => timeline.observations.filter(o => o.frame_index === frame), [timeline, frame]);
  const selectedObservation = observations.find(o => o.track_id === selected);
  const selectedTrack = timeline.tracks.find(track => track.id === selected);
  useEffect(() => { setDraftBox(selectedObservation?.bbox ? {...selectedObservation.bbox} : null); }, [selected, frame, selectedObservation?.id]);
  useEffect(() => { setSourceDraft(selectedTrack?.source_text || ""); setTargetDraft(selectedTrack?.target_text || ""); }, [selectedTrack?.id, selectedTrack?.revision]);
  const seek = useCallback((next: number) => { const value = Math.max(0, Math.min(max, next)); setFrame(value); if (video.current) { const absolute=pts[value]??value/fps, base=showLocalized?(pts[previewStart]??previewStart/fps):0; video.current.currentTime=Math.max(0,absolute-base); } }, [max, pts, fps, showLocalized, previewStart]);
  useEffect(() => { if (video.current) seek(showLocalized ? previewStart : frame); }, [showLocalized]);
  const toggle = () => { const el = video.current; if (!el) return; if (el.paused) void el.play().catch(e => setError(String(e))); else el.pause(); };
  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.matches("input,select,textarea,button")) return;
      if (event.code === "Space") { event.preventDefault(); toggle(); }
      if (event.key === "ArrowLeft") { event.preventDefault(); seek(frame - 1); }
      if (event.key === "ArrowRight") { event.preventDefault(); seek(frame + 1); }
    };
    window.addEventListener("keydown", keyboard); return () => window.removeEventListener("keydown", keyboard);
  }, [frame, seek]);

  const saveKeyframe = async () => {
    if (!selected) return;
    if (scope === "range" && (rangeStart === null || rangeEnd === null)) { setError("Mark both range boundaries before applying a range edit."); return; }
    setSaving(true); setError(null);
    try {
      await onKeyframe({ track_id: selected, frame_index: scope === "range" ? Math.min(rangeStart!, rangeEnd!) : frame,
        end_frame: scope === "range" ? Math.max(rangeStart!, rangeEnd!) : undefined, scope, bbox: draftBox || selectedObservation?.bbox });
      cache.current.clear(); setScopePrompt(false); requestSeq.current++; setTimeline(EMPTY);
      const value = await loadTimeline(job.id, Math.max(0, frame - 300), Math.min(max, frame + 300)); setTimeline(value);
    } catch (e) { setError(String(e)); } finally { setSaving(false); }
  };
  const saveTrack = async () => {
    if (!selected || !onTrackUpdate) return; setTrackSaving(true); setError(null);
    try { await onTrackUpdate(selected, { source_text: sourceDraft, target_text: targetDraft, status: targetDraft.trim() ? "translated" : "review_required", expected_revision: selectedTrack?.revision }); cache.current.clear(); setTimeline(await loadTimeline(job.id, Math.max(0, frame-300), Math.min(max, frame+300))); }
    catch(e){ setError(String(e)); } finally { setTrackSaving(false); }
  };
  const moveOverlay = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (!drag.current || !videoRect.width || !videoRect.height) return;
    const dx = (event.clientX-drag.current.x)*job.manifest.width/videoRect.width, dy=(event.clientY-drag.current.y)*job.manifest.height/videoRect.height;
    setDraftBox({...drag.current.box,x:Math.max(0,Math.min(job.manifest.width-drag.current.box.width,drag.current.box.x+dx)),y:Math.max(0,Math.min(job.manifest.height-drag.current.box.height,drag.current.box.y+dy))});
  };

  if (job.upgrade_required) return <div className="video-job-state" role="status"><AlertTriangle size={20}/><div><strong>Analysis upgrade required</strong><p>Reviewed track text is preserved; prototype observations and renders must be regenerated by the production pipeline.</p></div><div className="video-job-actions"><button disabled={!onUpgrade} onClick={()=>void onUpgrade?.()}><RotateCcw size={14}/> Upgrade analysis</button></div></div>;
  if (!ready) return <div className="video-job-state" role="status">
    {terminalFailure ? <AlertTriangle size={20}/> : <span className="video-job-spinner"/>}
    <div><strong>{job.status === "failed" ? "Video processing failed" : job.status === "cancelled" ? "Video processing cancelled" : `Preparing video: ${job.stage}`}</strong><p>{job.error || `${Math.round(job.progress * 100)}% complete`}</p></div>
    <div className="video-job-actions">{!terminalFailure && onCancel && <button onClick={() => void onCancel()}>Cancel</button>}{terminalFailure && onResume && <button onClick={() => void onResume()}><RotateCcw size={14}/> Resume</button>}</div>
  </div>;

  return <div className="video-workspace">
    {error && <div className="video-error" role="alert"><AlertTriangle size={14}/><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError(null)}><X size={14}/></button></div>}
    <div ref={stage} className="video-stage"><video ref={video} src={activeVideoUrl} preload="metadata" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
      <div className="video-overlays" style={videoRect}>{observations.map((o: TrackObservation) => { const box=selected===o.track_id&&draftBox?draftBox:o.bbox; return <button key={o.id} aria-label={`Select and move track ${o.track_id}, confidence ${Math.round(o.tracking_confidence*100)} percent`} onClick={() => setSelected(o.track_id)} onPointerDown={event=>{if(selected===o.track_id){event.currentTarget.setPointerCapture(event.pointerId);drag.current={x:event.clientX,y:event.clientY,box:{...box}}}}} onPointerMove={moveOverlay} onPointerUp={()=>{drag.current=null;if(selected===o.track_id)setScopePrompt(true)}} className={selected === o.track_id ? "selected editable" : ""} style={{ left:`${100*box.x/job.manifest.width}%`,top:`${100*box.y/job.manifest.height}%`,width:`${100*box.width/job.manifest.width}%`,height:`${100*box.height/job.manifest.height}%` }} />})}</div>
    </div>
    <div className="video-controls"><button aria-label="Previous frame" onClick={() => seek(frame-1)}><ChevronLeft size={16}/></button><button aria-label={playing?"Pause":"Play"} onClick={toggle}>{playing?<Pause size={16}/>:<Play size={16}/>}</button><button aria-label="Next frame" onClick={() => seek(frame+1)}><ChevronRight size={16}/></button><span className="video-timecode">{timecode(second)} · frame {frame}/{max}</span><button onClick={() => setRangeStart(frame)}>Mark in{rangeStart === null ? "" : `: ${rangeStart}`}</button><button onClick={() => setRangeEnd(frame)}>Mark out{rangeEnd === null ? "" : `: ${rangeEnd}`}</button><button disabled={!selected} onClick={() => setScopePrompt(true)}><Flag size={14}/> Edit keyframe</button><button disabled={!onPreview||job.status==="rendering"} onClick={()=>{const start=rangeStart??frame,end=rangeEnd??Math.min(max,frame+150);setPreviewStart(Math.min(start,end));void onPreview!(Math.min(start,end),Math.max(start,end))}}>Render preview</button><button disabled={!job.preview_url} onClick={()=>setShowLocalized(value=>!value)}>{showLocalized?"Show source":"Show localized"}</button><button disabled={!onExport||job.status==="rendering"} onClick={()=>void onExport!()}>Export MP4</button>{job.export_url&&<a className="video-download" href={job.export_url} download>Download export</a>}</div>
    <input aria-label="Video frame" aria-valuetext={`${timecode(second)}, frame ${frame}`} className="video-scrubber" type="range" min={0} max={max} value={frame} onChange={e => seek(Number(e.target.value))}/>
    <div className="video-timeline" aria-busy={loading}>{loading && <span className="video-timeline-loading">Loading timeline…</span>}{timeline.tracks.map((track: TextTrack) => <button key={track.id} onClick={() => { setSelected(track.id); seek(track.start_frame); }} className={selected===track.id?"selected":""}><span className="video-track-label"><strong>{track.target_text || track.source_text || track.id}</strong><small>{track.status}</small></span><span className="video-track-rail"><i style={{left:`${100*track.start_frame/Math.max(1,max)}%`,width:`${100*(track.end_frame-track.start_frame+1)/Math.max(1,max)}%`}} />{timeline.keyframes.filter(k=>k.track_id===track.id).map((k,i)=><b key={i} title={`Keyframe ${k.frame_index}`} style={{left:`${100*k.frame_index/Math.max(1,max)}%`}} />)}{timeline.issues.filter(issue=>issue.track_id===track.id&&issue.frame_index!==undefined).map(issue=><em key={issue.id} title={issue.detail||issue.code} style={{left:`${100*(issue.frame_index||0)/Math.max(1,max)}%`}} />)}</span></button>)}</div>
    {selectedTrack && <aside className="video-track-inspector"><header><div><strong>Track text</strong><small>{selectedTrack.shot_id} · frames {selectedTrack.start_frame}–{selectedTrack.end_frame} · OCR {Math.round(selectedTrack.consensus_confidence*100)}%</small></div><button aria-label="Close track inspector" onClick={()=>setSelected(null)}><X size={14}/></button></header><label>Source text<textarea value={sourceDraft} onChange={e=>setSourceDraft(e.target.value)}/></label><label>Translation<textarea value={targetDraft} onChange={e=>setTargetDraft(e.target.value)} placeholder="Enter localized text"/></label><footer><span>{draftBox?`Geometry ${Math.round(draftBox.x)}, ${Math.round(draftBox.y)} · ${Math.round(draftBox.width)}×${Math.round(draftBox.height)}`:"No observation on this frame"}</span><button disabled={trackSaving||!onTrackUpdate} onClick={()=>void saveTrack()}>{trackSaving?"Saving…":"Save track text"}</button></footer></aside>}
    {scopePrompt && <div className="video-scope-backdrop" onClick={() => setScopePrompt(false)}><div className="video-scope-dialog" role="dialog" aria-modal="true" aria-labelledby="video-scope-title" onClick={e=>e.stopPropagation()}><h3 id="video-scope-title">Apply this edit where?</h3><p>The current geometry will be stored as a render keyframe.</p><div>{(["frame","range","track"] as Scope[]).map(value=><label key={value}><input type="radio" name="scope" value={value} checked={scope===value} onChange={()=>setScope(value)}/>{value==="frame"?"Current frame":value==="range"?`Selected range${rangeStart===null||rangeEnd===null?" (mark in/out first)":` (${Math.min(rangeStart,rangeEnd)}–${Math.max(rangeStart,rangeEnd)})`}`:"Whole track"}</label>)}</div><footer><button onClick={()=>setScopePrompt(false)}>Cancel</button><button disabled={saving || (scope==="range"&&(rangeStart===null||rangeEnd===null))} onClick={()=>void saveKeyframe()}>{saving?"Saving…":"Apply edit"}</button></footer></div></div>}
  </div>;
}
