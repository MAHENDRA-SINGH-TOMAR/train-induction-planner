"""
AI-Driven Train Induction Planning & Scheduling - FastAPI Backend
==================================================================
Provides RESTful APIs and an Operations Research (Google OR-Tools MIP)
optimization engine for metro/railway train induction and stabling yard management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import copy
from ortools.linear_solver import pywraplp

app = FastAPI(
    title="RailOptix - Train Induction Planning API",
    description="AI-Driven Operations Research & Telemetry Engine for Train Induction Planning",
    version="2.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# DOMAIN DATA MODELS
# -------------------------------------------------------------
class OptimizationWeights(BaseModel):
    health: float = Field(default=35.0, ge=0.0, le=100.0, description="Weight for telemetry health score")
    shunting: float = Field(default=30.0, ge=0.0, le=100.0, description="Weight for shunting cost penalty")
    mileage: float = Field(default=20.0, ge=0.0, le=100.0, description="Weight for mileage equalization")
    cleaning: float = Field(default=15.0, ge=0.0, le=100.0, description="Weight for cleanliness & branding")

class DisruptionPayload(BaseModel):
    train_id: Optional[str] = "TS-108"
    defect_type: Optional[str] = "Brake Pressure Sensor Anomaly (>1.2 bar delta)"

class StandbyTogglePayload(BaseModel):
    train_id: str

# -------------------------------------------------------------
# INITIAL DEPOT STATE & ROLLING STOCK FLEET
# -------------------------------------------------------------
DEFAULT_TRACKS = [
    {"id": "S-01", "name": "Stabling Track 1", "capacity": 3, "trains": ["TS-101", "TS-102", "TS-103"]},
    {"id": "S-02", "name": "Stabling Track 2", "capacity": 3, "trains": ["TS-104", "TS-105", "TS-106"]},
    {"id": "S-03", "name": "Stabling Track 3", "capacity": 3, "trains": ["TS-107", "TS-108", "TS-109"]},
    {"id": "S-04", "name": "Stabling Track 4", "capacity": 3, "trains": ["TS-110", "TS-111", "TS-112"]},
    {"id": "IBL-1", "name": "Inspection Shed 1", "capacity": 2, "trains": ["TS-113", "TS-114"]},
    {"id": "IBL-2", "name": "Inspection Shed 2", "capacity": 2, "trains": ["TS-115", "TS-116"]},
    {"id": "WP-01", "name": "Automated Wash", "capacity": 2, "trains": ["TS-117", "TS-118"]}
]

DEFAULT_TRAINSETS = [
    {"id": "TS-101", "name": "Rake 101", "track": "S-01", "pos": 0, "status": "inducted", "slot": "05:15", "line": "Line 1 (Red)", "health": 98.0, "mileage": 82000, "clean": 95.0, "shunts": 0, "issues": "None", "brake_pad_mm": 24.2, "hvac_temp_c": 21.4},
    {"id": "TS-102", "name": "Rake 102", "track": "S-01", "pos": 1, "status": "inducted", "slot": "05:45", "line": "Line 1 (Red)", "health": 94.0, "mileage": 91000, "clean": 90.0, "shunts": 1, "issues": "None", "brake_pad_mm": 22.8, "hvac_temp_c": 21.6},
    {"id": "TS-103", "name": "Rake 103", "track": "S-01", "pos": 2, "status": "inducted", "slot": "06:15", "line": "Line 1 (Red)", "health": 91.0, "mileage": 84000, "clean": 92.0, "shunts": 2, "issues": "None", "brake_pad_mm": 21.5, "hvac_temp_c": 22.0},
    {"id": "TS-104", "name": "Rake 104", "track": "S-02", "pos": 0, "status": "inducted", "slot": "05:30", "line": "Line 2 (Blue)", "health": 96.0, "mileage": 78000, "clean": 98.0, "shunts": 0, "issues": "None", "brake_pad_mm": 25.0, "hvac_temp_c": 21.0},
    {"id": "TS-105", "name": "Rake 105", "track": "S-02", "pos": 1, "status": "inducted", "slot": "06:00", "line": "Line 2 (Blue)", "health": 89.0, "mileage": 89000, "clean": 88.0, "shunts": 1, "issues": "None", "brake_pad_mm": 20.4, "hvac_temp_c": 22.1},
    {"id": "TS-106", "name": "Rake 106", "track": "S-02", "pos": 2, "status": "inducted", "slot": "06:30", "line": "Line 2 (Blue)", "health": 88.0, "mileage": 95000, "clean": 85.0, "shunts": 2, "issues": "Minor door sensor jitter", "brake_pad_mm": 19.8, "hvac_temp_c": 22.4},
    {"id": "TS-107", "name": "Rake 107", "track": "S-03", "pos": 0, "status": "inducted", "slot": "06:45", "line": "Line 1 (Red)", "health": 95.0, "mileage": 76000, "clean": 90.0, "shunts": 0, "issues": "None", "brake_pad_mm": 24.0, "hvac_temp_c": 21.2},
    {"id": "TS-108", "name": "Rake 108", "track": "S-03", "pos": 1, "status": "inducted", "slot": "07:00", "line": "Line 2 (Blue)", "health": 90.0, "mileage": 83000, "clean": 87.0, "shunts": 1, "issues": "None", "brake_pad_mm": 21.0, "hvac_temp_c": 21.8},
    {"id": "TS-109", "name": "Rake 109", "track": "S-03", "pos": 2, "status": "standby", "slot": "HOT-STANDBY", "line": "Depot Throat", "health": 97.0, "mileage": 80500, "clean": 95.0, "shunts": 0, "issues": "Hot reserve for morning peak", "brake_pad_mm": 25.1, "hvac_temp_c": 20.9},
    {"id": "TS-110", "name": "Rake 110", "track": "S-04", "pos": 0, "status": "inducted", "slot": "07:15", "line": "Line 1 (Red)", "health": 92.0, "mileage": 87000, "clean": 89.0, "shunts": 0, "issues": "None", "brake_pad_mm": 22.0, "hvac_temp_c": 21.5},
    {"id": "TS-111", "name": "Rake 111", "track": "S-04", "pos": 1, "status": "inducted", "slot": "07:30", "line": "Line 2 (Blue)", "health": 93.0, "mileage": 85000, "clean": 91.0, "shunts": 1, "issues": "None", "brake_pad_mm": 23.4, "hvac_temp_c": 21.3},
    {"id": "TS-112", "name": "Rake 112", "track": "S-04", "pos": 2, "status": "inducted", "slot": "07:45", "line": "Line 1 (Red)", "health": 87.0, "mileage": 93500, "clean": 86.0, "shunts": 2, "issues": "None", "brake_pad_mm": 19.5, "hvac_temp_c": 22.6},
    {"id": "TS-113", "name": "Rake 113", "track": "IBL-1", "pos": 0, "status": "standby", "slot": "COLD-STANDBY", "line": "Depot Reserve", "health": 95.0, "mileage": 75000, "clean": 94.0, "shunts": 1, "issues": "Secondary cold reserve", "brake_pad_mm": 24.8, "hvac_temp_c": 21.1},
    {"id": "TS-114", "name": "Rake 114", "track": "IBL-1", "pos": 1, "status": "maintenance", "slot": "MAINTENANCE", "line": "Depot IBL", "health": 65.0, "mileage": 99400, "clean": 70.0, "shunts": 0, "issues": "Traction Inverter B Overhaul", "brake_pad_mm": 16.2, "hvac_temp_c": 24.0},
    {"id": "TS-115", "name": "Rake 115", "track": "IBL-2", "pos": 0, "status": "defective", "slot": "GROUNDED", "line": "Depot IBL", "health": 42.0, "mileage": 88000, "clean": 60.0, "shunts": 0, "issues": "Emergency Brake Cylinder Leak", "brake_pad_mm": 12.0, "hvac_temp_c": 26.5},
    {"id": "TS-116", "name": "Rake 116", "track": "IBL-2", "pos": 1, "status": "maintenance", "slot": "MAINTENANCE", "line": "Depot IBL", "health": 70.0, "mileage": 98000, "clean": 65.0, "shunts": 1, "issues": "Pantograph carbon strip replacement", "brake_pad_mm": 18.0, "hvac_temp_c": 23.0},
    {"id": "TS-117", "name": "Rake 117", "track": "WP-01", "pos": 0, "status": "defective", "slot": "GROUNDED", "line": "Wash Track", "health": 50.0, "mileage": 89000, "clean": 40.0, "shunts": 0, "issues": "HVAC Unit #3 Compressor Locked", "brake_pad_mm": 14.5, "hvac_temp_c": 28.2},
    {"id": "TS-118", "name": "Rake 118", "track": "WP-01", "pos": 1, "status": "maintenance", "slot": "WASH_IN_PROG", "line": "Wash Track", "health": 85.0, "mileage": 86000, "clean": 50.0, "shunts": 1, "issues": "Undergoing exterior bio-wash", "brake_pad_mm": 21.0, "hvac_temp_c": 22.0}
]

PEAK_SLOTS = [
    {"slot": "05:15", "line": "Line 1 (Red)"},
    {"slot": "05:30", "line": "Line 2 (Blue)"},
    {"slot": "05:45", "line": "Line 1 (Red)"},
    {"slot": "06:00", "line": "Line 2 (Blue)"},
    {"slot": "06:15", "line": "Line 1 (Red)"},
    {"slot": "06:30", "line": "Line 2 (Blue)"},
    {"slot": "06:45", "line": "Line 1 (Red)"},
    {"slot": "07:00", "line": "Line 2 (Blue)"},
    {"slot": "07:15", "line": "Line 1 (Red)"},
    {"slot": "07:30", "line": "Line 2 (Blue)"},
    {"slot": "07:45", "line": "Line 1 (Red)"}
]

# State storage
yard_state = {
    "tracks": copy.deepcopy(DEFAULT_TRACKS),
    "trainsets": copy.deepcopy(DEFAULT_TRAINSETS),
    "weights": OptimizationWeights(),
    "last_rationale": "Initial baseline induction schedule loaded. 11 rakes inducted for Morning Peak."
}

# -------------------------------------------------------------
# AI OPTIMIZATION & SCORING LOGIC
# -------------------------------------------------------------
def calculate_train_fitness(train: dict, weights: OptimizationWeights, target_mileage: float = 80000.0) -> float:
    """Calculates multi-objective fitness score (0-100) using normalized metrics."""
    if train["status"] == "defective" or train["health"] < 70.0:
        return 0.0

    # Shunting score: 0 shunts = 100, 1 shunt = 75, 2 shunts = 45
    shunt_score = 100.0 if train["shunts"] == 0 else (75.0 if train["shunts"] == 1 else 45.0)

    # Mileage equalization: deviation from target
    mileage_score = max(20.0, 100.0 - abs(train["mileage"] - target_mileage) / 250.0)

    total_weight = weights.health + weights.shunting + weights.mileage + weights.cleaning
    if total_weight <= 0:
        total_weight = 1.0

    fitness = (
        (train["health"] * weights.health) +
        (shunt_score * weights.shunting) +
        (mileage_score * weights.mileage) +
        (train["clean"] * weights.cleaning)
    ) / total_weight

    return round(fitness, 1)


def solve_induction_mip(trains: List[dict], slots: List[dict], weights: OptimizationWeights):
    """
    Mixed-Integer Linear Programming (MIP) using Google OR-Tools Solver (CBC/SCIP).
    Maximizes sum of fitness scores while respecting 1-train-per-slot and 1-slot-per-train.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        # Fallback to standard CBC or GLOP
        solver = pywraplp.Solver.CreateSolver("CBC")

    # Filter eligible candidates
    eligible_trains = [t for t in trains if t["status"] != "defective" and t["health"] >= 75.0]
    num_trains = len(eligible_trains)
    num_slots = len(slots)

    if not solver or num_trains < num_slots:
        # Fallback greedy ranking
        eligible_trains.sort(key=lambda t: calculate_train_fitness(t, weights), reverse=True)
        return {
            "assignments": [
                {"slot": slots[i]["slot"], "line": slots[i]["line"], "train_id": eligible_trains[i]["id"]}
                for i in range(min(num_slots, len(eligible_trains)))
            ],
            "solver_status": "GREEDY_FALLBACK"
        }

    # Decision variables: x[i, j] = 1 if train i assigned to slot j
    x = {}
    for i in range(num_trains):
        for j in range(num_slots):
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    # Constraint 1: Each slot must be assigned exactly one train
    for j in range(num_slots):
        solver.Add(solver.Sum(x[i, j] for i in range(num_trains)) == 1)

    # Constraint 2: Each train assigned to at most one slot
    for i in range(num_trains):
        solver.Add(solver.Sum(x[i, j] for j in range(num_slots)) <= 1)

    # Objective: Maximize total fitness with early slot bonus for 0-shunt trains
    objective = solver.Objective()
    for i in range(num_trains):
        base_fitness = calculate_train_fitness(eligible_trains[i], weights)
        for j in range(num_slots):
            # Trains with 0 shunting get a slight multiplier for earlier slots to exit smoothly
            slot_bonus = 2.0 if (eligible_trains[i]["shunts"] == 0 and j < 4) else 0.0
            coef = base_fitness + slot_bonus
            objective.SetCoefficient(x[i, j], coef)
    objective.SetMaximization()

    status = solver.Solve()
    assignments = []
    assigned_ids = set()

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for j in range(num_slots):
            for i in range(num_trains):
                if x[i, j].solution_value() > 0.5:
                    t_id = eligible_trains[i]["id"]
                    assignments.append({
                        "slot": slots[j]["slot"],
                        "line": slots[j]["line"],
                        "train_id": t_id
                    })
                    assigned_ids.add(t_id)
        solver_status_text = "OPTIMAL_OR_TOOLS_MIP"
    else:
        # Fallback greedy
        eligible_trains.sort(key=lambda t: calculate_train_fitness(t, weights), reverse=True)
        for i in range(num_slots):
            t_id = eligible_trains[i]["id"]
            assignments.append({
                "slot": slots[i]["slot"],
                "line": slots[i]["line"],
                "train_id": t_id
            })
            assigned_ids.add(t_id)
        solver_status_text = "GREEDY_FALLBACK"

    return {
        "assignments": assignments,
        "solver_status": solver_status_text,
        "assigned_ids": assigned_ids
    }

# -------------------------------------------------------------
# REST API ENDPOINTS
# -------------------------------------------------------------
@app.get("/api/status")
def get_status():
    """Returns overview KPIs and system operational status."""
    trains = yard_state["trainsets"]
    inducted = sum(1 for t in trains if t["status"] == "inducted")
    standby = sum(1 for t in trains if t["status"] == "standby")
    maint = sum(1 for t in trains if t["status"] == "maintenance")
    defective = sum(1 for t in trains if t["status"] == "defective")
    shunts_total = sum(t["shunts"] for t in trains if t["status"] == "inducted")

    return {
        "status": "OPERATIONAL",
        "depot_id": "EY-04",
        "depot_name": "East Yard Main Maintenance Depot",
        "induction_window": "05:00 - 09:30",
        "total_fleet": len(trains),
        "inducted_count": inducted,
        "standby_count": standby,
        "maintenance_count": maint,
        "defective_count": defective,
        "shunts_saved": max(0, 22 - shunts_total),
        "last_rationale": yard_state["last_rationale"]
    }

@app.get("/api/fleet")
def get_fleet():
    """Returns the full fleet of trainsets with fitness calculations and telemetry."""
    weights = yard_state["weights"]
    result = []
    for t in yard_state["trainsets"]:
        item = dict(t)
        item["fitness_score"] = calculate_train_fitness(t, weights)
        result.append(item)
    return result

@app.get("/api/tracks")
def get_tracks():
    """Returns depot yard tracks, capacities, and train occupancy."""
    return yard_state["tracks"]

@app.get("/api/schedule")
def get_schedule():
    """Returns the scheduled timetable of inducted trainsets."""
    weights = yard_state["weights"]
    inducted = [t for t in yard_state["trainsets"] if t["status"] == "inducted"]
    inducted.sort(key=lambda x: x["slot"])
    
    schedule = []
    for t in inducted:
        schedule.append({
            "slot": t["slot"],
            "line": t["line"],
            "train_id": t["id"],
            "track": t["track"],
            "shunts": t["shunts"],
            "shunting_latency_min": t["shunts"] * 4,
            "health": t["health"],
            "fitness_score": calculate_train_fitness(t, weights)
        })
    return schedule

@app.post("/api/optimize")
def optimize_schedule(weights: OptimizationWeights):
    """
    Triggers Google OR-Tools MIP solver to generate the optimal train induction schedule
    balancing telemetry health, shunting minimization, mileage equalization, and cleaning.
    """
    yard_state["weights"] = weights
    trains = yard_state["trainsets"]

    sol = solve_induction_mip(trains, PEAK_SLOTS, weights)
    assignments = {item["slot"]: (item["train_id"], item["line"]) for item in sol["assignments"]}
    assigned_train_ids = {item["train_id"] for item in sol["assignments"]}

    # Update statuses
    for t in trains:
        if t["id"] in assigned_train_ids:
            t["status"] = "inducted"
            for s, (tid, line) in assignments.items():
                if tid == t["id"]:
                    t["slot"] = s
                    t["line"] = line
                    break
        elif t["status"] not in ["maintenance", "defective"]:
            # Assign hot standby to best remaining
            if "TS-109" == t["id"] or "TS-113" == t["id"]:
                t["status"] = "standby"
                t["slot"] = "HOT-STANDBY"
                t["line"] = "Depot Throat"
            else:
                t["status"] = "standby"
                t["slot"] = "COLD-STANDBY"
                t["line"] = "Depot Reserve"

    yard_state["last_rationale"] = (
        f"AI ({sol['solver_status']}) optimized induction with weights: "
        f"Health {weights.health}%, Shunting {weights.shunting}%, "
        f"Mileage {weights.mileage}%, Cleaning {weights.cleaning}%. "
        f"{len(assignments)} peak rakes scheduled with minimized bottleneck shunting."
    )

    return {
        "success": True,
        "solver": sol["solver_status"],
        "assigned_count": len(assignments),
        "schedule": get_schedule(),
        "rationale": yard_state["last_rationale"]
    }

@app.post("/api/disrupt")
def simulate_disruption(payload: DisruptionPayload):
    """
    Simulates rolling stock failure during pre-induction, triggering automatic
    standby rake substitution and dynamic timetable re-allocation.
    """
    trains = yard_state["trainsets"]
    target = next((t for t in trains if t["id"] == payload.train_id), None)
    if not target:
        target = next((t for t in trains if t["status"] == "inducted"), None)

    if not target:
        raise HTTPException(status_code=400, detail="No suitable inducted train found to disrupt.")

    vacated_slot = target["slot"]
    vacated_line = target["line"]
    target["status"] = "defective"
    target["health"] = 35.0
    target["slot"] = "GROUNDED"
    target["issues"] = payload.defect_type or "Emergency Brake Differential Pressure Alarm"

    # Find standby candidate
    standby = next((t for t in trains if t["status"] == "standby" and t["health"] >= 80.0), None)
    if standby:
        standby["status"] = "inducted"
        standby["slot"] = vacated_slot
        standby["line"] = vacated_line
        rationale = (
            f"DISRUPTION MITIGATED: {target['id']} grounded ({target['issues']}). "
            f"AI dynamic swap assigned Hot Standby {standby['id']} to slot {vacated_slot} "
            f"on {vacated_line} with 0 min departure delay."
        )
        yard_state["last_rationale"] = rationale
        return {
            "success": True,
            "disrupted": target["id"],
            "replacement": standby["id"],
            "slot": vacated_slot,
            "rationale": rationale
        }
    else:
        rationale = f"ALERT: {target['id']} grounded. No ready standby rake available. Headway adjusted."
        yard_state["last_rationale"] = rationale
        return {
            "success": False,
            "disrupted": target["id"],
            "replacement": None,
            "rationale": rationale
        }

@app.post("/api/trains/{train_id}/standby")
def toggle_standby(train_id: str):
    """Toggles a trainset between Hot Standby and Inducted/Ready."""
    target = next((t for t in yard_state["trainsets"] if t["id"] == train_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Trainset not found.")

    if target["status"] == "standby":
        target["status"] = "inducted"
        target["slot"] = "08:00"
        target["line"] = "Line 1 (Red)"
    else:
        target["status"] = "standby"
        target["slot"] = "HOT-STANDBY"
        target["line"] = "Depot Throat"

    return {"success": True, "train": target}

@app.post("/api/reset")
def reset_state():
    """Resets yard and trainsets to default operational state."""
    yard_state["tracks"] = copy.deepcopy(DEFAULT_TRACKS)
    yard_state["trainsets"] = copy.deepcopy(DEFAULT_TRAINSETS)
    yard_state["weights"] = OptimizationWeights()
    yard_state["last_rationale"] = "Depot state reset to baseline schedule."
    return {"success": True, "message": "State reset"}

@app.get("/healthz")
def healthz():
    """Healthcheck endpoint for Render / cloud deployment monitoring."""
    return {"status": "healthy", "service": "railoptix-train-induction"}

# -------------------------------------------------------------
# STATIC FRONTEND MOUNTING
# -------------------------------------------------------------
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting RailOptix server on {host}:{port}...")
    uvicorn.run("server:app", host=host, port=port, reload=False)
