import React from "react";

/**
 * EDIT THIS FILE to change the taxonomy shown at the bottom of the ER diagram page.
 *
 *  - departments: the list of municipal departments.
 *  - taxonomy: each category with its subcategories. Every subcategory is
 *    ["Subcategory name", "Department name"] and the department must appear in `departments`.
 *
 * The page updates automatically; no other file needs to change.
 */

const departments = [
  "Roads Department",
  "Drainage Department",
  "Waste Management Department",
  "Sanitation Department",
  "Water Supply Department",
  "Sewerage Department",
  "Street Lighting Department",
  "Electrical Department",
  "Pollution Control Department",
] as const;

type Department = typeof departments[number];

interface TaxonomyCategory {
  category: string;
  subs: [string, Department][];
}

const taxonomy: TaxonomyCategory[] = [
  {
    category: "Roads & Footpaths",
    subs: [
      ["Potholes & Road Damage", "Roads Department"],
      ["Road Repair / Construction", "Roads Department"],
      ["Footpath Damage / Repair", "Roads Department"],
      ["Road Waterlogging", "Drainage Department"],
    ],
  },
  {
    category: "Waste & Sanitation",
    subs: [
      ["Garbage Collection / Removal", "Waste Management Department"],
      ["Garbage Dumping", "Waste Management Department"],
      ["Street / Public Area Cleaning", "Sanitation Department"],
      ["Waste Disposal & Segregation", "Waste Management Department"],
    ],
  },
  {
    category: "Water Supply",
    subs: [
      ["Water Supply Interruption", "Water Supply Department"],
      ["Water Leakage", "Water Supply Department"],
      ["Water Pipeline Damage / Maintenance", "Water Supply Department"],
      ["New Water Connection / Pipeline", "Water Supply Department"],
    ],
  },
  {
    category: "Sewage & Drainage",
    subs: [
      ["Sewer Blockage / Overflow", "Sewerage Department"],
      ["Sewer Line Repair / Maintenance", "Sewerage Department"],
      ["Drain Blockage / Desilting", "Drainage Department"],
      ["Storm Water Drainage / Waterlogging", "Drainage Department"],
    ],
  },
  {
    category: "Street Lighting & Electricity",
    subs: [
      ["Street Light Not Working / Repair", "Street Lighting Department"],
      ["Street Light Installation", "Street Lighting Department"],
      ["Electrical Pole / Transformer Issues", "Electrical Department"],
      ["Electricity Supply", "Electrical Department"],
    ],
  },
  {
    category: "Pollution",
    subs: [
      ["Air Pollution", "Pollution Control Department"],
      ["Noise Pollution", "Pollution Control Department"],
      ["Water Pollution", "Pollution Control Department"],
      ["Land / Waste Pollution", "Pollution Control Department"],
    ],
  },
];

const LIGHT_THEME = {
  background: "#ffffff",
  textMain: "#111827",
  textMuted: "#4b5563",
  border: "#e5e7eb",
  primaryHeaderBg: "#1f2937", // Crisp dark charcoal header for high contrast & elegance
  primaryHeaderText: "#ffffff",
  tableHeaderBg: "#f9fafb",
  badgeBg: "#f3f4f6",
  badgeBorder: "#d1d5db",
  badgeText: "#374151",
};

export default function TaxonomySection() {
  return (
    <div style={{ background: LIGHT_THEME.background, color: LIGHT_THEME.textMain, padding: "24px 24px" }}>
      <h1 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 6px" }}>Complaint taxonomy</h1>
      <p style={{ margin: "0 0 20px", fontSize: 13, color: LIGHT_THEME.textMuted, lineHeight: 1.5 }}>
        Final categories, subcategories, and departments. These rows fill the categories
        and subcategories tables, and each subcategory maps directly to its responsible department.
      </p>

      {/* Departments Section */}
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8, color: LIGHT_THEME.textMain }}>
        Departments ({departments.length})
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 28 }}>
        {departments.map((dept) => (
          <span
            key={dept}
            style={{
              border: `1px solid ${LIGHT_THEME.badgeBorder}`,
              borderRadius: 999,
              padding: "5px 14px",
              fontSize: 13,
              background: LIGHT_THEME.badgeBg,
              color: LIGHT_THEME.badgeText,
              fontWeight: 500,
            }}
          >
            {dept}
          </span>
        ))}
      </div>

      {/* Taxonomy Grid */}
      <div
        style={{
          display: "grid",
          gap: 16,
          gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
        }}
      >
        {taxonomy.map((cat, index) => (
          <div
            key={cat.category}
            style={{
              border: `1px solid ${LIGHT_THEME.border}`,
              borderRadius: 8,
              overflow: "hidden",
              background: "#ffffff",
              boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
            }}
          >
            <div
              style={{
                background: LIGHT_THEME.primaryHeaderBg,
                color: LIGHT_THEME.primaryHeaderText,
                padding: "10px 14px",
                fontWeight: 700,
                fontSize: 14,
              }}
            >
              {index + 1}. {cat.category}
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ background: LIGHT_THEME.tableHeaderBg, textAlign: "left" }}>
                  <th style={{ padding: "8px 14px", fontWeight: 600, color: LIGHT_THEME.textMain }}>
                    Subcategory
                  </th>
                  <th style={{ padding: "8px 14px", fontWeight: 600, color: LIGHT_THEME.textMain }}>
                    Department
                  </th>
                </tr>
              </thead>
              <tbody>
                {cat.subs.map(([sub, dept]) => (
                  <tr key={sub} style={{ borderTop: `1px solid ${LIGHT_THEME.border}` }}>
                    <td style={{ padding: "9px 14px", color: LIGHT_THEME.textMain }}>{sub}</td>
                    <td style={{ padding: "9px 14px", color: LIGHT_THEME.textMuted }}>{dept}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}