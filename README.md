# AI-Driven Train Induction Planning & Scheduling

An intelligent, multi-objective decision support system and user interface designed for metro and rail depot controllers, operations planners, and rolling stock engineers.

---

## What is Train Induction Planning?

Train induction is the critical daily operational workflow where trainsets (EMUs/rakes) stabled in the maintenance yard are selected, inspected, prioritized, and dispatched to the mainline to meet the morning and evening revenue peak timetables.

Sub-optimal induction leads to:
* **Yard Neck Bottlenecks & Shunting Delays**: Trapped trains requiring multiple shunt moves cause timetable delays before service even starts.
* **Premature Overhaul / Wear Clustering**: Over-utilizing certain rakes causes maintenance overhauls to bunch up together, creating fleet availability crises.
* **Rolling Stock Breakdown in Revenue Service**: Inducting trains with borderline sensor telemetry causes mainline stall events and system-wide delays.

---

## Key Features of the UI & Engine

### 1. Multi-Objective AI Fitness Scoring
Evaluates and ranks all available trainsets based on 4 customizable weighted criteria:
$$\text{Fitness Score} = w_{\text{health}} \cdot H + w_{\text{shunting}} \cdot S + w_{\text{mileage}} \cdot M + w_{\text{cleaning}} \cdot C$$

* **Telemetric Health & Diagnostics ($H$)**: Real-time IoT sensor telemetry (brake cylinder pressures, traction motor inverter temperatures, HVAC performance, battery float voltage, pantograph carbon strip wear).
* **Shunting & Yard Geometry Cost ($S$)**: Minimizes unnecessary shunting moves based on physical track occupancy and depot throat access.
* **Fleet Mileage Equalization ($M$)**: Balances cumulative rake kilometers to evenly stagger scheduled preventive overhauls.
* **Cleanliness & Branding Compliance ($C$)**: Validates deep sanitization sign-off and commercial advertisement wrapping contracts.

### 2. Interactive Depot Stabling & Shunting Visualizer
* Displays depot stabling roads (S-01 to S-04), Inspection Bay Lines (IBL-1, IBL-2), and Wash Plants (WP-01).
* Identifies trains that have direct throat exit vs. trains blocked behind other rakes.
* Color-coded status: **Inducted (Emerald)**, **Hot/Cold Standby (Cyan)**, **Maintenance (Amber)**, **Grounded / Defect (Rose)**.

### 3. Explainable AI (XAI) Recommendation Panel
* Provides natural language justifications for why specific rakes are selected for specific time slots.
* Dynamic sliders allow operators to adjust objective priorities in real-time.

### 4. Dynamic Disruption Simulation & Auto-Mitigation
* One-click "Simulate Rake Defect" triggers an emergency defect on an inducted train.
* The AI engine immediately locates the optimal Hot Standby rake, reassigns the departure slot, recalculates shunting routing, and minimizes departure delay.

### 5. Detailed Rolling Stock Telemetry Modal
* Click any trainset in the yard or schedule to inspect component-level health, brake pad thickness, HVAC temperatures, and open work orders.

---

## Architecture

The system consists of a **FastAPI backend** running a **Google OR-Tools MIP Optimization Solver** serving REST endpoints and the responsive Tailwind CSS UI.

```
       ┌────────────────────────┐
       │   Frontend Web UI      │
       │ (Tailwind + Yard SVG)  │
       └───────────┬────────────┘
                   │  Fetch REST API
                   ▼
       ┌────────────────────────┐
       │     FastAPI Server     │  Port 3000
       │      (server.py)       │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   Google OR-Tools MIP  │
       │   SCIP/CBC Solver      │
       └────────────────────────┘
```

---

## REST API Endpoints

The server runs on `http://localhost:3000`. You can also explore the live interactive Swagger documentation at **[http://localhost:3000/docs](http://localhost:3000/docs)**.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Train Induction Dashboard |
| `GET` | `/api/status` | Current yard KPIs, total fleet, and induction window summary |
| `GET` | `/api/fleet` | Fleet rolling stock telemetry, health scores, and track coordinates |
| `GET` | `/api/tracks` | Depot track lines (`S-01` to `S-04`, `IBL-1/2`, `WP-01`) |
| `GET` | `/api/schedule` | Inducted train schedule sorted by departure slot |
| `POST` | `/api/optimize` | Runs Google OR-Tools MIP solver with custom objective weights |
| `POST` | `/api/disrupt` | Injects train defect and triggers dynamic standby rake reallocation |
| `POST` | `/api/trains/{id}/standby` | Toggles trainset Hot Standby designation |
| `POST` | `/api/reset` | Resets depot state back to default baseline |

### 2. Run the Python AI Engine Test
```bash
python C:\Users\HP\.gemini\antigravity\scratch\train-induction-planner\induction_engine.py
```
