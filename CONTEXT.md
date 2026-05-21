# CONTEXT.md — EuroMillions Crowd Analyser
## Full Project Brief for Claude Code Initialization

---

## What this project is

A full-stack data analysis tool for EuroMillions lottery strategy.

**Core insight:** Lottery draws are certified random — you cannot improve your
odds of winning. But you CAN reduce jackpot splits by picking combinations that
fewer humans choose. This tool models human cognitive bias in number selection
and helps generate, track, and analyse picks that avoid crowded combinations.

**Stack:** TypeScript (Next.js frontend + API routes) + Python (analysis engine,
prediction models, data processing) + React/JSX (UI components).

---

## Stack & Architecture

```
euromillions/
├── frontend/                  # Next.js 14 + TypeScript + Tailwind
│   ├── app/
│   │   ├── page.tsx           # Root — tab shell
│   │   ├── layout.tsx
│   │   └── api/
│   │       ├── generate/route.ts     # POST → calls Python engine
│   │       ├── analyze/route.ts      # POST → scores a combination
│   │       ├── predict/route.ts      # POST → prediction engine
│   │       ├── picks/route.ts        # GET/POST → pick history
│   │       ├── draws/route.ts        # GET → draw history
│   │       ├── import/route.ts       # POST → CSV upload
│   │       └── stats/route.ts        # GET → frequency statistics
│   ├── components/
│   │   ├── NumberBall.tsx
│   │   ├── StarBall.tsx
│   │   ├── UniquenessMeter.tsx
│   │   ├── HeatmapGrid.tsx
│   │   ├── FrequencyBar.tsx
│   │   ├── ScoreChart.tsx
│   │   └── tabs/
│   │       ├── GenerateTab.tsx
│   │       ├── AnalyzeTab.tsx
│   │       ├── HistoryTab.tsx
│   │       ├── PredictTab.tsx
│   │       ├── StatsTab.tsx
│   │       └── WhyTab.tsx
│   └── lib/
│       ├── types.ts           # Shared TS types
│       ├── bias.ts            # Client-side bias model (mirrors Python)
│       └── api.ts             # API client helpers
│
├── engine/                    # Python analysis package
│   ├── __init__.py
│   ├── bias.py                # Human bias model
│   ├── data.py                # Draw/pick schema, CSV parser, statistics
│   ├── generator.py           # Weighted pick generation
│   ├── predictor.py           # 10-model prediction engine
│   ├── report.py              # HTML dashboard builder
│   └── server.py              # FastAPI server (called by Next.js API routes)
│
├── data/                      # Persisted JSON (gitignored)
│   ├── my_picks.json
│   ├── draw_history.json
│   ├── computed_stats.json
│   └── predictions.json
│
├── requirements.txt
├── package.json
├── tsconfig.json
└── CONTEXT.md                 # ← this file
```

---

## Core Domain Logic

### 1. Human Bias Popularity Model (`engine/bias.py` + `frontend/lib/bias.ts`)

Every number 1–50 and star 1–12 has a **popularity score (10–100)**.
Higher = more humans pick it = more jackpot splits if it wins.

**Bias rules (both Python and TypeScript must implement identically):**

| Rule | Affected | Delta | Reason |
|---|---|---|---|
| Birthday Effect | 1–31 | +35 | People pick dates |
| Month Bonus | 1–12 | +15 | Also valid as months |
| Low Number Gravity | 1–10 | +18 | Cognitive pull to small numbers |
| Lucky 7 Family | 7,17,27,37,47 | +20 | Universally "lucky" digit |
| Lucky 7 Apex | 7 | +15 | Extra boost for #7 itself |
| Lucky 3 Family | 3,13,23,33,43 | +8 | Secondary lucky digit |
| Round Numbers | 10,20,30,40,50 | +12 | Feel organised |
| Fibonacci | 1,2,3,5,8,21,34 | +12 | Mathematical attraction |
| High Neglect | 40–50 | -12 | Outside birthday range |
| Very High Neglect | 45–50 | -8 | Extra neglect |
| Unlucky 13 Avoidance | 13 | -25 | Superstition — contrarian edge |

Baseline = 50. Clamp result to [10, 100].

**Stars (1–12):**

| Rule | Affected | Delta |
|---|---|---|
| Low Star Bias | 1–6 | +30 |
| Very Low Star | 1–3 | +20 |
| Lucky Star 7 | 7 | +15 |
| High Star Neglect | 9–12 | -20 |
| Very High Neglect | 11–12 | -12 |

**Uniqueness Score** = `round(100 - mean(all 7 popularity scores))`
Range 0–100. Higher = fewer splits if this wins.

**Tier labels:**
- 85–100: Very Crowded 🔴
- 70–84: Crowded 🟠
- 50–69: Average ⬜
- 35–49: Underplayed 🟢
- 10–34: Rare Pick 🔵

---

### 2. Weighted Pick Generation (`engine/generator.py`)

Uses **inverse-popularity weighting** — numbers with lower popularity score
are proportionally more likely to be selected.

```python
weight(n) = 1.0 / popularity(n)
# Sample without replacement, k=5 for numbers, k=2 for stars
```

Optional **blend mode**: blend draw frequency into bias weights.
If a number is drawn less than expected (cold), reduce its popularity score
by up to 10 points, making it even more likely to be selected.

---

### 3. Prediction Engine (`engine/predictor.py`)

10 individual models, each scoring numbers 0.0–1.0:

| Model | Window | Logic |
|---|---|---|
| `frequency_regression` | last 50 draws | Numbers drawn below expected frequency |
| `overdue` | all draws | Gap since last appearance (normalised to 20-draw max) |
| `hot_streak` | last 20 draws | Numbers above expected frequency |
| `sum_proximity` | all draws | Target: historical mean sum ≈ 125–130 |
| `pair_frequency` | last 100 draws | Co-occurrence pairs |
| `recency_decay` | all draws | Exponential decay weight (λ=0.92) per draw back |
| `odd_even_balance` | last 50 | Target historical odd/even ratio |
| `decade_balance` | last 50 | Spread across 1-10, 11-20, 21-30, 31-40, 41-50 |
| `consecutive_gap` | last 50 | Historical frequency of consecutive numbers |
| `star_overdue` | all draws | Gap since each star last appeared |

**Default ensemble weights:**
```python
{
  "frequency_regression": 0.20,
  "overdue":              0.20,
  "hot_streak":           0.10,
  "recency_decay":        0.25,
  "pair_frequency":       0.25,
}
```

Weights are configurable — users should tune and track accuracy to learn
which (if any) models show signal above chance.

**Prediction pick generation:**
- Score each number by ensemble
- Add floor of 0.05 so all numbers have some probability
- Weighted sample without replacement
- Filter/rank by sum proximity to historical mean

**IMPORTANT:** Every prediction output must include the disclaimer:
> "EuroMillions draws are certified random. These models detect retrospective patterns, not future outcomes. Track accuracy to measure performance — expect chance-level results."

---

### 4. Data Schema (TypeScript types in `frontend/lib/types.ts`)

```typescript
interface Draw {
  date: string;           // ISO date string "2024-03-15"
  numbers: number[];      // sorted, length 5, range 1-50
  stars: number[];        // sorted, length 2, range 1-12
  jackpot?: number;       // in euros
  winners?: number;
}

interface Pick {
  id: string;
  timestamp: string;      // ISO datetime
  numbers: number[];
  stars: number[];
  uniquenessScore: number; // 0-100
  source: "generated" | "manual" | "predicted";
  note?: string;
}

interface Prediction extends Pick {
  sum: number;
  sumTarget: number;
  modelsUsed: string[];
  explanation: {
    numbers: Array<{ number: number; signal: number; reasons: string[] }>;
    stars:   Array<{ star: number; signal: number; reasons: string[] }>;
  };
}

interface DrawStats {
  totalDraws: number;
  dateRange: { first: string; last: string };
  numFrequency: Record<number, number>;
  starFrequency: Record<number, number>;
  expectedNumFreq: number;
  expectedStarFreq: number;
  hotNumbers: number[];
  coldNumbers: number[];
  hotStars: number[];
  coldStars: number[];
  topPairs: Array<{ pair: [number, number]; count: number }>;
  topStarPairs: Array<{ pair: [number, number]; count: number }>;
  sumStats: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
  };
}
```

---

### 5. Python FastAPI Server (`engine/server.py`)

Next.js API routes call this Python server for heavy computation.

```
POST /generate          { count: number, blendDraws: boolean }
POST /analyze           { numbers: number[], stars: number[] }
POST /predict           { count: number }
GET  /picks             → Pick[]
POST /picks             Pick (save)
GET  /draws             → Draw[]
POST /import            multipart CSV upload
GET  /stats             → DrawStats
GET  /accuracy          → match distribution
```

---

## UI Components

### Tab Structure

```
[ Generate ] [ Analyze ] [ History ] [ Predict ] [ Stats ] [ Why ]
```

#### Generate Tab (`GenerateTab.tsx`)
- Button: "Generate Picks" with optional count (1–10)
- Toggle: "Blend draw history" (calls /generate?blend=true)
- Shows grid of picks, sorted by uniqueness score descending
- Each pick: number balls + star balls + uniqueness meter
- Best pick has a "★ BEST" badge
- Per-number bias breakdown table for top pick
- All generated picks auto-saved to history

#### Analyze Tab (`AnalyzeTab.tsx`)
- Interactive 50-number grid (click to select, max 5)
- Interactive 12-star grid (click to select, max 2)
- Balls coloured by bias tier (red → blue)
- Live uniqueness score updates as numbers selected
- Per-number bias factor breakdown on submit
- Option to save to history

#### History Tab (`HistoryTab.tsx`)
- Table: date, numbers, stars, score, source, best draw match
- Sortable by date / score
- Score timeline chart (canvas)
- Best match column compares each pick against all draws
  - Format: "3n+1s (2024-02-09)" in green if ≥3 matches

#### Predict Tab (`PredictTab.tsx`)
- Requires 50+ draws imported
- Shows disclaimer panel (yellow, non-dismissable)
- Top 10 number signals table (signal, overdue, cold, recency, hot columns)
- Top 4 star signals
- Generated picks with per-number reasoning
- Accuracy tracker: match distribution vs actual subsequent draws

#### Stats Tab (`StatsTab.tsx`)
- Requires draw history
- Frequency bar chart (all 50 numbers)
- Hot/cold number badges
- Star frequency breakdown
- Sum distribution histogram
- Top pairs table
- Date range and total draws shown

#### Why Tab (`WhyTab.tsx`)
- 6 accordion cards, one per bias factor
- Each shows: icon, name, affected range, impact badge, description
- Collapsible with expand/collapse animation
- Disclaimer at bottom

---

### Key UI Components

#### `NumberBall.tsx`
```tsx
interface NumberBallProps {
  n: number;
  size?: number;         // default 40
  showTier?: boolean;    // show "Underplayed" label below
  selected?: boolean;
  onClick?: () => void;
}
// Border colour = popColor(NUMBER_POPULARITY[n])
// Background = dark (#0a1528)
// Shape = circle
```

#### `StarBall.tsx`
```tsx
// Same as NumberBall but rotated 45deg (diamond shape)
// Border colour = popColor(STAR_POPULARITY[s])
```

#### `UniquenessMeter.tsx`
```tsx
interface UniquenessMeterProps {
  score: number;          // 0-100
  showLabel?: boolean;
}
// Gradient fill bar: score/100 width
// Colour: green ≥65, yellow ≥45, red <45
// Label: "Split-Avoidance Score  {score}/100 · {tier}"
```

#### `HeatmapGrid.tsx`
```tsx
interface HeatmapGridProps {
  mode: "bias" | "frequency";
  data: Record<number, number>;  // number → score or frequency
  maxVal?: number;
  onHover?: (n: number) => void;
  size?: 50 | 12;   // main numbers or stars
}
// 10-column grid for size=50, 6-column for size=12
// Cell colour interpolated from score/maxVal
```

---

## Design System

**Aesthetic:** Dark data terminal. Clinical, analytical — like a quant's tool, not a gambling app.

**Colours (CSS variables):**
```css
--bg:      #05080f
--bg1:     #0a1020
--bg2:     #0d1528
--border:  #1a2540
--text:    #d0d8f0
--dim:     #4a5a7a
--accent:  #4fc3f7   /* cyan */
--green:   #7ed4a4
--yellow:  #f0c040
--red:     #ff6b6b
--orange:  #ff9a3c
```

**Fonts:**
- Display/headings: `Syne` (800 weight for title)
- Mono/data: `JetBrains Mono` (numbers, scores, labels)
- Body: `DM Sans` (descriptions, prose)

**Component style principles:**
- Balls have coloured borders matching bias tier, dark fill
- Stars are rotated 45° squares (diamond shape)
- Meters are thin (4–6px) gradient bars with glow
- Cards: `bg1` background, `border` border, 12px radius
- All popups/tooltips: `bg2` with `border2`
- No shadows except subtle `box-shadow: 0 0 12px {color}44` on selected balls

---

## Data Files (gitignore these)

```
data/my_picks.json         # { id, timestamp, numbers, stars, uniquenessScore, source }[]
data/draw_history.json     # { date, numbers, stars, jackpot?, winners? }[]
data/computed_stats.json   # full DrawStats object
data/predictions.json      # Prediction[] with explanation
```

CSV sources for draw history:
- UK: `https://www.national-lottery.co.uk/results/euromillions/draw-history/csv`
- FR: `https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/historique`

Supported CSV column formats (parser handles all):
```
Draw Date / Ball 1-5 / Lucky Star 1-2
date / n1-n5 / s1-s2
drawdate / ball1-5 / luckystar1-2
```

---

## Python Package (`engine/`)

Install:
```bash
pip install -r requirements.txt
# or
pip install fastapi uvicorn numpy pandas scipy scikit-learn matplotlib seaborn rich click requests beautifulsoup4 python-dateutil tabulate
```

Run server:
```bash
uvicorn engine.server:app --reload --port 8001
```

---

## NPM / TypeScript Setup

```bash
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend
npm install recharts lucide-react @radix-ui/react-tabs @radix-ui/react-tooltip
```

---

## Claude Code Instructions

When initialising this project, Claude Code should:

1. **Read this entire file first** before touching any code
2. Create the full directory structure as shown above
3. Implement `bias.py` and `bias.ts` identically — same numbers, same weights
4. Implement `types.ts` exactly as defined — all API routes use these types
5. The Python FastAPI server is the source of truth for computation;
   TypeScript only does UI rendering and lightweight client-side bias scoring
6. Every prediction output **must** include the randomness disclaimer
7. Data files live in `/data/` at project root, not inside frontend or engine
8. Use `fetch('/api/...')` from components — never call Python server directly from client
9. The uniqueness score formula is `round(100 - mean([...numPops, ...starPops]))`
   — implement this identically in both Python and TypeScript

### Priority build order:
1. `engine/bias.py` → `frontend/lib/bias.ts` (core model, everything depends on this)
2. `engine/data.py` (schema + CSV parser)
3. `engine/generator.py`
4. `engine/server.py` (FastAPI with all routes)
5. `frontend/lib/types.ts`
6. `frontend/components/NumberBall.tsx`, `StarBall.tsx`, `UniquenessMeter.tsx`
7. `frontend/app/api/*/route.ts` (proxy routes to Python)
8. Tab components in order: Generate → Analyze → History → Stats → Predict → Why
9. `engine/predictor.py` (most complex — build last)
10. `engine/report.py` (standalone HTML dashboard, optional if full Next.js app exists)

---

## What was built before this project

This project supersedes two earlier prototypes:

1. **React artifact** (`euromillions-crowd-avoider.jsx`) — standalone JSX with
   all logic inline. The bias model and UI design from this artifact should be
   the reference for component aesthetics and the generate/analyze/why tabs.

2. **Python CLI** (`euromillions_project.zip`) — standalone Python with
   `bias.py`, `data.py`, `generator.py`, `predictor.py`. This is the reference
   implementation for all analysis logic. Port the core algorithms exactly.

The new project unifies both: Python engine for computation, TypeScript/React
for UI, proper project structure with shared types and API layer.

---

## Disclaimer (must appear in UI and all prediction outputs)

> EuroMillions draws are independently certified as truly random using hardware RNG.
> No statistical model can predict lottery outcomes. This tool analyses human
> selection behaviour to minimise jackpot splits — not to improve winning odds.
> Track prediction accuracy over time to observe chance-level performance.
