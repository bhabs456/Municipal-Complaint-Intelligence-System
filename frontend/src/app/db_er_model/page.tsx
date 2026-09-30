"use client"
import React from "react";

/**
 * Municipal Complaint Intelligence System - ER Diagram
 * 8 tables, crow's-foot notation, plain SVG (no dependencies besides React).
 */

type Key = "PK" | "FK" | "UQ" | "";
type Field = { key: Key; name: string; type: string; note?: string };
type Table = {
  id: string;
  x: number;
  y: number;
  w: number;
  color: string;
  core?: boolean;
  fields: Field[];
};

const HEADER_H = 32;
const ROW_H = 22;

const INK = "#1f2a37";
const MUTED = "#6b7683";
const LINE = "#3d4b5c";

const tables: Table[] = [
  {
    id: "users", x: 50, y: 40, w: 270, color: "#3d4b5c",
    fields: [
      { key: "PK", name: "user_id", type: "INT" },
      { key: "", name: "name", type: "VARCHAR(100)" },
      { key: "UQ", name: "email", type: "VARCHAR(150)" },
      { key: "", name: "phone", type: "VARCHAR(15)" },
      { key: "", name: "password_hash", type: "VARCHAR(255)" },
      { key: "", name: "role", type: "ENUM" , note: "CITIZEN / ADMIN / OFFICER"},
      { key: "", name: "created_at", type: "DATETIME" },
    ],
  },
  {
    id: "complaints", x: 460, y: 40, w: 280, color: "#0f5c5c", core: true,
    fields: [
      { key: "PK", name: "complaint_id", type: "INT" },
      { key: "FK", name: "user_id", type: "INT" },
      { key: "", name: "full_complaint", type: "TEXT", note: "raw, never overwritten" },
      { key: "", name: "title", type: "VARCHAR(200)" },
      { key: "", name: "description", type: "TEXT" },
      { key: "", name: "contact_number", type: "VARCHAR(15)" },
      { key: "", name: "issue_datetime", type: "DATETIME" },
      { key: "", name: "submitted_at", type: "DATETIME" },
      { key: "", name: "status", type: "VARCHAR(30)" },
      { key: "", name: "updated_at", type: "DATETIME" },
    ],
  },
  {
    id: "complaint_locations", x: 900, y: 40, w: 270, color: "#4a6a8a",
    fields: [
      { key: "PK", name: "location_id", type: "INT" },
      { key: "FK", name: "complaint_id", type: "INT", note: "UNIQUE" },
      { key: "", name: "address", type: "VARCHAR(255)" },
      { key: "", name: "latitude", type: "DECIMAL(9,6)" },
      { key: "", name: "longitude", type: "DECIMAL(9,6)" },
      { key: "", name: "landmark", type: "VARCHAR(150)" },
      { key: "", name: "location_source", type: "ENUM" },
      { key: "", name: "created_at", type: "DATETIME" },
    ],
  },
  {
    id: "complaint_analysis", x: 460, y: 400, w: 280, color: "#6b3f7a",
    fields: [
      { key: "PK", name: "analysis_id", type: "INT" },
      { key: "FK", name: "complaint_id", type: "INT", note: "UNIQUE" },
      { key: "", name: "is_civic", type: "BOOLEAN" },
      { key: "FK", name: "category_id", type: "INT" },
      { key: "FK", name: "subcategory_id", type: "INT" },
      { key: "", name: "confidence_score", type: "FLOAT" },
      { key: "", name: "duplicate_score", type: "FLOAT" },
      { key: "FK", name: "duplicate_complaint_id", type: "INT", note: "nullable" },
      { key: "", name: "seriousness_score", type: "FLOAT" },
      { key: "", name: "location_impact_score", type: "FLOAT" },
      { key: "", name: "complaint_density_score", type: "FLOAT" },
      { key: "", name: "priority_score", type: "FLOAT" },
      { key: "", name: "priority_level", type: "VARCHAR(20)" },
      { key: "", name: "analyzed_at", type: "DATETIME" },
      { key: "", name: "model_version", type: "VARCHAR(30)" },
    ],
  },
  {
    id: "categories", x: 900, y: 380, w: 270, color: "#9a6a12",
    fields: [
      { key: "PK", name: "category_id", type: "INT" },
      { key: "", name: "category_name", type: "VARCHAR(100)" },
      { key: "", name: "description", type: "TEXT" },
    ],
  },
  {
    id: "subcategories", x: 900, y: 580, w: 270, color: "#9a6a12",
    fields: [
      { key: "PK", name: "subcategory_id", type: "INT" },
      { key: "FK", name: "category_id", type: "INT" },
      { key: "", name: "subcategory_name", type: "VARCHAR(100)" },
      { key: "", name: "department", type: "VARCHAR(100)" },
    ],
  },
  {
    id: "complaint_interactions", x: 50, y: 360, w: 270, color: "#4a6a8a",
    fields: [
      { key: "PK", name: "interaction_id", type: "INT" },
      { key: "FK", name: "complaint_id", type: "INT" },
      { key: "FK", name: "user_id", type: "INT" },
      { key: "", name: "interaction_type", type: "ENUM" },
      { key: "", name: "created_at", type: "DATETIME" },
    ],
  },
  {
    id: "complaint_history", x: 50, y: 580, w: 270, color: "#4a6a8a",
    fields: [
      { key: "PK", name: "history_id", type: "INT" },
      { key: "FK", name: "complaint_id", type: "INT" },
      { key: "", name: "status", type: "VARCHAR(30)" },
      { key: "", name: "assigned_department", type: "VARCHAR(100)" },
      { key: "", name: "remarks", type: "TEXT" },
      { key: "FK", name: "changed_by", type: "INT" },
      { key: "", name: "created_at", type: "DATETIME" },
    ],
  },
];

const significance: Record<string, string> = {
  users:
    "Stores everyone who uses the system: citizens who report problems, admins who manage the platform, and municipal officers who resolve issues. The role column controls what each person can do.",
  complaints:
    "The central table. It holds the complaint exactly as the citizen submitted it, including the untouched full_complaint text, and the current status. Every other table hangs off this one.",
  complaint_locations:
    "Keeps the geographic details (address, latitude, longitude, landmark) apart from the complaint text. The coordinates support map views, complaint density, heatmaps and location-based duplicate checks.",
  complaint_analysis:
    "Holds everything the NLP and ML pipeline produces for a complaint: civic or non-civic, category, subcategory, confidence, duplicate score, seriousness, location impact, density and the final priority score and level. The original complaint is never modified.",
  categories:
    "A reference list of the main municipal complaint types, such as Roads & Footpaths, Water Supply or Waste & Sanitation. It keeps classification results consistent.",
  subcategories:
    "A reference list of specific issues under each category, such as Pothole under Roads & Footpaths. Each subcategory also names its responsible department, so the department is looked up rather than predicted by the model.",
  complaint_interactions:
    "Records community actions on a complaint: upvotes, duplicate confirmations and follows. A user can support a given complaint only once, and these signals adjust attention after the initial priority is set.",
  complaint_history:
    "An append-only log of the complaint's lifecycle, from Submitted through Under Review, Assigned, In Progress and Resolved to Closed. Each change records the status, department, remarks and who made it, so nothing is overwritten.",
};

type Dir = [number, number]; // unit vector pointing away from the entity
type End = { dir: Dir; kind: "one" | "many" };
type Rel = {
  ids: [string, string];
  points: [number, number][];
  from: End;
  to: End;
  dashed?: boolean;
};

const R: Dir = [1, 0];
const L: Dir = [-1, 0];
const D: Dir = [0, 1];
const U: Dir = [0, -1];

const relations: Rel[] = [
  // Users
  { ids: ["users", "complaints"], points: [[320, 105], [460, 105]], from: { dir: R, kind: "one" }, to: { dir: L, kind: "many" } },
  { ids: ["users", "complaint_interactions"], points: [[175, 226], [175, 360]], from: { dir: D, kind: "one" }, to: { dir: U, kind: "many" } },
  { ids: ["users", "complaint_history"], points: [[50, 60], [25, 60], [25, 733], [50, 733]], from: { dir: L, kind: "one" }, to: { dir: L, kind: "many" } },
  // Complaints
  { ids: ["complaints", "complaint_locations"], points: [[740, 105], [900, 105]], from: { dir: R, kind: "one" }, to: { dir: L, kind: "one" } },
  { ids: ["complaints", "complaint_analysis"], points: [[540, 292], [540, 400]], from: { dir: D, kind: "one" }, to: { dir: U, kind: "one" } },
  { ids: ["complaints", "complaint_interactions"], points: [[460, 250], [380, 250], [380, 425], [320, 425]], from: { dir: L, kind: "one" }, to: { dir: R, kind: "many" } },
  { ids: ["complaints", "complaint_history"], points: [[460, 275], [400, 275], [400, 645], [320, 645]], from: { dir: L, kind: "one" }, to: { dir: R, kind: "many" } },
  // Duplicate link (analysis -> complaints)
  { ids: ["complaint_analysis", "complaints"], points: [[690, 400], [690, 292]], from: { dir: U, kind: "many" }, to: { dir: D, kind: "one" }, dashed: true },
  // Reference tables
  { ids: ["categories", "subcategories"], points: [[1035, 478], [1035, 580]], from: { dir: D, kind: "one" }, to: { dir: U, kind: "many" } },
  { ids: ["categories", "complaint_analysis"], points: [[900, 423], [820, 423], [820, 509], [740, 509]], from: { dir: L, kind: "one" }, to: { dir: R, kind: "many" } },
  { ids: ["subcategories", "complaint_analysis"], points: [[900, 623], [860, 623], [860, 531], [740, 531]], from: { dir: L, kind: "one" }, to: { dir: R, kind: "many" } },
];

const ACCENT = "#c2410c";

function EndLabel({ x, y, end, color }: { x: number; y: number; end: End; color: string }) {
  const [dx, dy] = end.dir;
  const text = end.kind === "one" ? "1" : "N";
  const horizontal = dx !== 0;
  const tx = horizontal ? x + dx * 14 : x + 11;
  const ty = horizontal ? y - 7 : y + dy * 16 + 4;
  return (
    <text
      x={tx} y={ty} textAnchor="middle" fontSize={14} fontWeight={800}
      fill={color} fontFamily="'Segoe UI', Arial, sans-serif"
      stroke="#fff" strokeWidth={3} paintOrder="stroke"
    >
      {text}
    </text>
  );
}

function Marker({ x, y, end, color }: { x: number; y: number; end: End; color: string }) {
  const [dx, dy] = end.dir;
  const px = -dy;
  const py = dx;
  const bar = (t: number) => (
    <line
      x1={x + dx * t + px * 7} y1={y + dy * t + py * 7}
      x2={x + dx * t - px * 7} y2={y + dy * t - py * 7}
      stroke={color} strokeWidth={1.8}
    />
  );
  if (end.kind === "one") {
    return <g>{bar(9)}{bar(15)}</g>;
  }
  const fx = x + dx * 15;
  const fy = y + dy * 15;
  return (
    <g stroke={color} strokeWidth={1.8} fill="none">
      <line x1={fx} y1={fy} x2={x + px * 8} y2={y + py * 8} />
      <line x1={fx} y1={fy} x2={x - px * 8} y2={y - py * 8} />
      <line x1={fx} y1={fy} x2={x} y2={y} />
      {bar(20)}
    </g>
  );
}

function Relation({
  rel, active, dimmed, showSymbols,
}: { rel: Rel; active: boolean; dimmed: boolean; showSymbols: boolean }) {
  const pts = rel.points;
  const first = pts[0];
  const last = pts[pts.length - 1];
  const color = active ? ACCENT : LINE;
  return (
    <g style={{ opacity: dimmed ? 0.15 : 1, transition: "opacity 0.2s" }}>
      <polyline
        points={pts.map((p) => p.join(",")).join(" ")}
        fill="none" stroke={color} strokeWidth={active ? 3 : 1.6}
        strokeDasharray={rel.dashed ? "6 4" : undefined}
        strokeLinejoin="round"
      />
      {showSymbols && <Marker x={first[0]} y={first[1]} end={rel.from} color={color} />}
      {showSymbols && <Marker x={last[0]} y={last[1]} end={rel.to} color={color} />}
      <EndLabel x={first[0]} y={first[1]} end={rel.from} color={color} />
      <EndLabel x={last[0]} y={last[1]} end={rel.to} color={color} />
    </g>
  );
}

type Highlight = "selected" | "connected" | "dimmed" | "none";

function TableBox({
  t, state, onSelect,
}: { t: Table; state: Highlight; onSelect: (id: string) => void }) {
  const h = HEADER_H + t.fields.length * ROW_H;
  return (
    <g
      fontFamily="'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
      style={{ cursor: "pointer", opacity: state === "dimmed" ? 0.25 : 1, transition: "opacity 0.2s" }}
      onClick={(e) => { e.stopPropagation(); onSelect(t.id); }}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(t.id); } }}
      tabIndex={0}
      role="button"
      aria-pressed={state === "selected"}
      aria-label={`Table ${t.id}`}
    >
      {(state === "selected" || state === "connected") && (
        <rect
          x={t.x - 6} y={t.y - 6} width={t.w + 12} height={h + 12} rx={10}
          fill={state === "selected" ? "rgba(194,65,12,0.08)" : "none"}
          stroke={ACCENT} strokeWidth={state === "selected" ? 3 : 2}
          strokeDasharray={state === "connected" ? "6 4" : undefined}
        />
      )}
      <rect
        x={t.x} y={t.y} width={t.w} height={h} rx={6}
        fill="#ffffff" stroke={t.color} strokeWidth={t.core ? 3 : 1.5}
      />
      <path
        d={`M${t.x},${t.y + HEADER_H} V${t.y + 6} a6,6 0 0 1 6,-6 H${t.x + t.w - 6} a6,6 0 0 1 6,6 V${t.y + HEADER_H} Z`}
        fill={t.color}
      />
      <text
        x={t.x + t.w / 2} y={t.y + 21} textAnchor="middle"
        fill="#fff" fontSize={t.core ? 15 : 14} fontWeight={700} letterSpacing={0.3}
      >
        {t.id}
      </text>
      {t.fields.map((f, i) => {
        const y = t.y + HEADER_H + i * ROW_H;
        return (
          <g key={f.name}>
            {i > 0 && (
              <line x1={t.x + 1} x2={t.x + t.w - 1} y1={y} y2={y} stroke="#e4e8ee" />
            )}
            {f.key && (
              <text
                x={t.x + 10} y={y + 15} fontSize={10} fontWeight={700}
                fill={f.key === "PK" ? "#b4531a" : f.key === "FK" ? "#1d6fa5" : MUTED}
              >
                {f.key}
              </text>
            )}
            <text
              x={t.x + 38} y={y + 15} fontSize={12.5} fill={INK}
              fontWeight={f.key === "PK" ? 700 : 400}
              textDecoration={f.key === "PK" ? "underline" : undefined}
            >
              {f.name}
            </text>
            <text x={t.x + t.w - 10} y={y + 15} fontSize={10.5} fill={MUTED} textAnchor="end">
              {f.type}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export default function MunicipalComplaintERD() {
  const [selected, setSelected] = React.useState<string | null>(null);
  const [showSymbols, setShowSymbols] = React.useState(false);

  const onSelect = (id: string) => setSelected((cur) => (cur === id ? null : id));

  const connectedIds = React.useMemo(() => {
    const set = new Set<string>();
    if (!selected) return set;
    relations.forEach((r) => {
      if (r.ids[0] === selected) set.add(r.ids[1]);
      if (r.ids[1] === selected) set.add(r.ids[0]);
    });
    set.delete(selected);
    return set;
  }, [selected]);

  const tableState = (id: string): Highlight => {
    if (!selected) return "none";
    if (id === selected) return "selected";
    return connectedIds.has(id) ? "connected" : "dimmed";
  };

  return (
    <div
      style={{
        background: "#ffffff", color: INK, padding: "24px 16px",
        fontFamily: "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
      }}
    >
      <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700 }}>
        Municipal Complaint Intelligence System
      </h1>
      <p style={{ margin: "4px 0 12px", color: MUTED, fontSize: 14 }}>
        Entity-relationship diagram with 8 tables. Click a table to highlight what it connects to.
      </p>

      <div
        style={{
          display: "flex", flexWrap: "wrap", gap: 16, alignItems: "center",
          marginBottom: 12, fontSize: 14, minHeight: 24,
        }}
      >
        <span>
          {selected ? (
            <>
              <strong>{selected}</strong> connects to:{" "}
              {[...connectedIds].join(", ") || "nothing"}
            </>
          ) : (
            <span style={{ color: MUTED }}>No table selected</span>
          )}
        </span>
        {selected && (
          <button
            onClick={() => setSelected(null)}
            style={{
              border: `1px solid ${LINE}`, background: "#fff", borderRadius: 6,
              padding: "3px 10px", cursor: "pointer", fontSize: 13,
            }}
          >
            Clear selection
          </button>
        )}
        <label style={{ marginLeft: "auto", cursor: "pointer", fontSize: 13 }}>
          <input
            type="checkbox" checked={showSymbols}
            onChange={(e) => setShowSymbols(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          Also show crow&apos;s foot symbols
        </label>
      </div>

      <div style={{ overflowX: "auto" }}>
        <svg
          viewBox="0 0 1220 830" width="100%" style={{ minWidth: 960, maxWidth: 1400 }}
          role="group" aria-label="ER diagram of the Municipal Complaint Intelligence System"
          onClick={() => setSelected(null)}
        >
          {relations.map((r, i) => {
            const active = !!selected && r.ids.includes(selected);
            return (
              <Relation
                key={i} rel={r} active={active}
                dimmed={!!selected && !active} showSymbols={showSymbols}
              />
            );
          })}

          {tables.map((t) => (
            <TableBox key={t.id} t={t} state={tableState(t.id)} onSelect={onSelect} />
          ))}

          <g
            fontSize={11} fill={MUTED} fontFamily="'Segoe UI', Arial, sans-serif"
            style={{ opacity: selected ? 0.4 : 1 }}
          >
            <text x={700} y={350}>duplicate_complaint_id</text>
            <text x={700} y={364}>(nullable, optional)</text>
          </g>

          <g fontSize={11.5} fill={MUTED} fontFamily="'Segoe UI', Arial, sans-serif">
            <text x={50} y={522}>UNIQUE (complaint_id, user_id, interaction_type)</text>
            <text x={50} y={538}>allows one upvote per user per complaint</text>
            <text x={900} y={728}>department is fixed per subcategory,</text>
            <text x={900} y={744}>so the ML model never predicts it.</text>
          </g>

          <g transform="translate(900,775)" fontSize={11.5} fill={INK} fontFamily="'Segoe UI', Arial, sans-serif">
            <text x={0} y={0} fontWeight={700}>Legend</text>
            <text x={0} y={18}><tspan fontWeight={800}>1</tspan> = exactly one</text>
            <text x={0} y={34}><tspan fontWeight={800}>N</tspan> = many</text>
            <text x={130} y={18}>PK primary key</text>
            <text x={130} y={34}>FK foreign key</text>
            <text x={250} y={18}>Dashed line = optional link</text>
          </g>
        </svg>
      </div>

      <h2 style={{ fontSize: 18, margin: "28px 0 12px" }}>Why each table exists</h2>
      <div
        style={{
          display: "grid", gap: 12,
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        }}
      >
        {tables.map((t) => (
          <div
            key={t.id}
            onClick={() => onSelect(t.id)}
            style={{
              borderStyle: "solid",
              borderWidth: "1px 1px 1px 5px",
              borderColor: selected === t.id ? ACCENT : "#e4e8ee",
              borderLeftColor: selected === t.id ? ACCENT : t.color,
              borderRadius: 6, padding: "10px 14px", cursor: "pointer",
              background: selected === t.id ? "rgba(194,65,12,0.06)" : "#fff",
            }}
          >
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{t.id}</div>
            <div style={{ fontSize: 13, lineHeight: 1.5, color: "#3d4b5c" }}>
              {significance[t.id]}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}