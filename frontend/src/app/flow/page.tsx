"use client"

import { useState } from "react";
import { Check, ChevronDown } from "lucide-react";

type Status = "done" | "next" | "pending";
type Stage = { n: number; title: string; question: string; purpose: string; status: Status; phases: string[] };

// Each phase: "id|title|brief"
const stages: Stage[] = [
  {
    n: 1, title: "Research, planning & system design", status: "done",
    question: "What should we build and why?",
    purpose: "Understand the problem, study existing work, and design the solution, taxonomy, architecture and database.",
    phases: [
      "1.1|Project & problem definition|Defines the municipal complaint problem, objectives, scope and expected outcome.",
      "1.2|Literature review & existing systems|Studies research papers, municipal systems and enterprise ticketing tools.",
      "1.3|Dataset study & data requirements|Finds which datasets exist and which complaint fields the system needs.",
      "1.4|Complaint taxonomy design|Fixes the structure: 6 categories and 24 subcategories.",
      "1.5|Department mapping|Maps every subcategory to the department responsible for it.",
      "1.6|Classification approach|Designs the flow: civic or non-civic, then category, then subcategory.",
      "1.7|Duplicate / related detection|Designs how reports about the same issue are recognized.",
      "1.8|Community upvoting|Lets citizens back an existing complaint instead of filing another.",
      "1.9|Priority & decision support|Defines problem seriousness, location impact, complaint density and the priority score.",
      "1.10|System architecture & workflow|Shows how frontend, backend, database, ML and decision support interact.",
      "1.11|Database / ER design|Designs the 8-table database and its relationships.",
      "1.12|Research gap & novelty|States what makes this system different from existing approaches.",
      "1.13|Future modules|Lists features that can be added later.",
      "1.14|Implementation roadmap|Turns the design into development stages and phases.",
    ],
  },
  {
    n: 2, title: "Data & AI/ML development", status: "next",
    question: "How will the intelligence work?",
    purpose: "Prepare the complaint dataset, then build the classification, duplicate detection and priority components.",
    phases: [
      "2.1|Dataset inspection|Checks records, fields, missing values, duplicates and raw categories before changing anything.",
      "2.2|Data cleaning|Fixes or removes incorrect, incomplete and unusable records.",
      "2.3|Subcategory normalization|Collapses 220+ raw subcategories into the final 24.",
      "2.4|Final taxonomy mapping|Maps each complaint to category, subcategory and department.",
      "2.5|Civic / non-civic handling|Marks whether a complaint belongs to the municipal domain.",
      "2.6|Text preparation|Prepares complaint text for ML while keeping its meaning.",
      "2.7|Class distribution & imbalance|Counts records per class and handles underrepresented ones.",
      "2.8|Train / validation / test split|Splits data for fair evaluation, blocking duplicate and near-duplicate leakage.",
      "2.9|ML dataset creation|Produces the final training-ready dataset.",
      "2.10|Dataset verification|Verifies labels, mappings, missing values and splits.",
      "2.11|Civic / non-civic classifier|First model layer: is this a municipal complaint at all?",
      "2.12|Category classification|Trains a model for the 6 major categories.",
      "2.13|Subcategory classification|Trains category-aware models for the 24 subcategories.",
      "2.14|Confidence scoring|Measures how sure the model is about each prediction.",
      "2.15|ML model evaluation|Scores models on accuracy, precision, recall, F1 and confusion matrix.",
      "2.16|Duplicate / related detection|Combines text similarity, location, category match and time/status into one duplicate score.",
      "2.17|Priority engine|Scores urgency from problem seriousness, location impact and complaint density.",
      "2.18|Dynamic priority reassessment|Recalculates priority when new complaints or supporting information arrive.",
    ],
  },
  {
    n: 3, title: "System development & integration", status: "pending",
    question: "How will all components work together?",
    purpose: "Connect the AI/ML parts with the backend, database and interfaces to make one working application.",
    phases: [
      "3.1|Backend foundation|FastAPI, SQLAlchemy, PostgreSQL/PostGIS and the database models.",
      "3.2|Complaint APIs|Endpoints to submit, retrieve, update and track the status of complaints.",
      "3.3|ML backend integration|Connects the classification models to the API.",
      "3.4|Duplicate detection integration|Runs the duplicate check during complaint submission.",
      "3.5|Priority engine integration|Attaches a priority score to each analyzed complaint.",
      "3.6|Complaint lifecycle|Submitted, Under Review, Assigned, In Progress, Resolved, Closed.",
      "3.7|Citizen frontend|Submission, location, landmark, photo upload, duplicate suggestions, upvoting and tracking.",
      "3.8|Admin / municipal dashboard|Complaint management, department and priority views, status, statistics and workload.",
      "3.9|Geographic / heatmap interface|Maps complaints and shows where civic issues concentrate.",
      "3.10|End-to-end workflow|Citizen complaint, backend, classification, duplicates, priority, department, dashboard.",
    ],
  },
  {
    n: 4, title: "Testing, deployment & evaluation", status: "pending",
    question: "Does it work correctly and can we deploy it?",
    purpose: "Verify the whole system, deploy it, and judge whether it meets its goals.",
    phases: [
      "4.1|API testing|Tests every backend endpoint.",
      "4.2|Database testing|Checks relationships, constraints, queries and data integrity.",
      "4.3|ML testing|Evaluates classification on unseen test data.",
      "4.4|Duplicate detection testing|Measures how accurately duplicates and related complaints are found.",
      "4.5|Priority engine testing|Runs priority calculations against different scenarios.",
      "4.6|Frontend testing|Tests the citizen and admin interfaces.",
      "4.7|End-to-end testing|Follows a complaint from submission through AI analysis and admin action to resolution.",
      "4.8|Edge cases & error handling|Tries unusual, incomplete, ambiguous and incorrect inputs.",
      "4.9|Dockerization|Packages backend, database and app components with Docker.",
      "4.10|Deployment|Sets up the production environment and deploys the prototype.",
      "4.11|Performance evaluation|Measures response time, ML inference time, database speed and reliability.",
      "4.12|Final project evaluation|Compares the result with the original objectives and research gap.",
      "4.13|Documentation & final report|Architecture, methodology, results, findings, screenshots, limitations and future work.",
    ],
  },
];

const C = {
  paper: "#F2F5F1", ink: "#1D2B33", muted: "#5F6F78", rule: "#CBD4CF",
  done: "#2E7D5B", next: "#D18A00", pending: "#93A0A7", card: "#FFFFFF",
};
const label: Record<Status, string> = { done: "Completed", next: "Next up", pending: "Pending" };
const sans = "system-ui, sans-serif";

const phaseStatus = (s: Stage, i: number): Status =>
  s.status === "done" ? "done" : s.status === "next" && i === 0 ? "next" : "pending";

export default function MunicipalRoadmap() {
  const [active, setActive] = useState(2);
  const [open, setOpen] = useState<string | null>("2.1");
  const total = stages.reduce((a, s) => a + s.phases.length, 0);
  const doneCount = stages.filter((s) => s.status === "done").reduce((a, s) => a + s.phases.length, 0);
  const stage = stages[active - 1];

  return (
    <div style={{ background: C.paper, color: C.ink, fontFamily: "Georgia, 'Times New Roman', serif" }} className="min-h-screen w-full">
      <style>{`
        @keyframes drawLine { from { transform: scaleY(0); } to { transform: scaleY(1); } }
        @keyframes pulseRing { 0% { transform: scale(1); opacity: .55; } 100% { transform: scale(2); opacity: 0; } }
        @keyframes grow { from { width: 0; } }
        .mc-line { transform-origin: top; animation: drawLine 1s ease-out both; }
        .mc-ring { animation: pulseRing 1.8s ease-out infinite; }
        .mc-bar { animation: grow 1.2s ease-out both; }
        .mc-body { display: grid; transition: grid-template-rows .25s ease; }
        @media (prefers-reduced-motion: reduce) { .mc-line, .mc-ring, .mc-bar { animation: none; } .mc-body { transition: none; } }
      `}</style>

      <div className="max-w-3xl mx-auto px-5 py-10">
        <h1 className="text-3xl sm:text-4xl font-bold leading-tight">Municipal Complaint Intelligence System</h1>
        <p className="mt-2" style={{ color: C.muted, fontFamily: sans }}>
          Reads citizen complaints, routes them to the right department, merges duplicates and ranks what to fix first.
        </p>

        <div className="mt-6 p-4 rounded-lg" style={{ background: C.card, border: `1px solid ${C.rule}`, fontFamily: sans }}>
          <div className="flex flex-wrap justify-between gap-1 text-sm mb-2">
            <span>{doneCount} of {total} phases complete</span>
            <span className="font-semibold" style={{ color: C.next }}>Now: Stage 2 begins with 2.1 dataset inspection</span>
          </div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: C.rule }}>
            <div key={doneCount} className="mc-bar h-full rounded-full" style={{ width: `${(doneCount / total) * 100}%`, background: C.done }} />
          </div>
        </div>

        {/* Stage selector */}
        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-2" style={{ fontFamily: sans }}>
          {stages.map((s) => {
            const on = active === s.n;
            return (
              <button
                key={s.n}
                onClick={() => { setActive(s.n); setOpen(s.status === "next" ? `${s.n}.1` : null); }}
                className="text-left p-3 rounded-lg relative"
                style={{ background: on ? C.ink : C.card, color: on ? "#fff" : C.ink, border: `1px solid ${on ? C.ink : C.rule}`, transition: "background .2s" }}
              >
                {s.status === "next" && <span className="mc-ring absolute top-3 right-3 w-2.5 h-2.5 rounded-full" style={{ background: C.next }} />}
                <span className="block text-xs" style={{ opacity: 0.7 }}>Stage {s.n}</span>
                <span className="block text-sm font-semibold leading-snug mt-0.5">{s.title}</span>
                <span className="block text-xs mt-1 font-semibold" style={{ color: on ? "#fff" : C[s.status] }}>
                  {s.status === "done" ? "✓ " : ""}{label[s.status]}
                </span>
              </button>
            );
          })}
        </div>

        {/* Stage header */}
        <div className="mt-8">
          <h2 className="text-xl font-bold">Stage {stage.n}: {stage.title}</h2>
          <p className="mt-1 italic" style={{ color: C[stage.status] }}>“{stage.question}”</p>
          <p className="mt-1 text-sm" style={{ color: C.muted, fontFamily: sans }}>{stage.purpose}</p>
        </div>

        {/* Phase route */}
        <div key={stage.n} className="relative mt-5">
          {stage.phases.map((raw, i) => {
            const [id, title, brief] = raw.split("|");
            const st = phaseStatus(stage, i);
            const col = C[st];
            const isOpen = open === id;
            return (
              <div key={id} className="relative pl-10 pb-2">
                <div className="absolute left-0 top-3 w-6 h-6 flex items-center justify-center">
                  {st === "next" && <span className="mc-ring absolute w-6 h-6 rounded-full" style={{ background: col }} />}
                  <span className="relative w-6 h-6 rounded-full flex items-center justify-center"
                    style={{ background: st === "pending" ? C.paper : col, border: `3px solid ${col}` }}>
                    {st === "done" && <Check size={12} color="#fff" strokeWidth={3} />}
                  </span>
                </div>
                <div className="rounded-lg" style={{ background: C.card, border: `1px solid ${isOpen ? col : C.rule}`, transition: "border-color .2s" }}>
                  <button onClick={() => setOpen(isOpen ? null : id)} aria-expanded={isOpen} className="w-full flex items-center gap-3 px-4 py-3 text-left">
                    <span className="text-sm font-semibold" style={{ color: col, fontFamily: sans, minWidth: "2.5rem" }}>{id}</span>
                    <span className="flex-1 font-semibold">{title}</span>
                    <ChevronDown size={16} style={{ transform: isOpen ? "rotate(180deg)" : "none", transition: "transform .2s" }} />
                  </button>
                  <div className="mc-body" style={{ gridTemplateRows: isOpen ? "1fr" : "0fr" }}>
                    <div className="overflow-hidden">
                      <p className="px-4 pb-3 text-sm" style={{ color: C.muted, fontFamily: sans }}>{brief}</p>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Overall flow */}
        <div className="mt-8 flex flex-wrap items-center gap-2 text-xs" style={{ fontFamily: sans }}>
          {stages.map((s, i) => (
            <span key={s.n} className="flex items-center gap-2">
              <button onClick={() => { setActive(s.n); setOpen(null); }} className="px-2.5 py-1 rounded-full"
                style={{ background: s.status === "pending" ? "transparent" : C[s.status], color: s.status === "pending" ? C.muted : "#fff", border: `1px solid ${s.status === "pending" ? C.rule : "transparent"}` }}>
                {["Research + design", "Data + AI/ML", "Build + integrate", "Test + deploy"][i]}
              </button>
              {i < 3 && <span style={{ color: C.pending }}>›</span>}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}