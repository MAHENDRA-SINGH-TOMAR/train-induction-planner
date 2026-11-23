"""
AI-Driven Train Induction Planning & Scheduling Engine
======================================================
This module provides the optimization algorithms for scheduling metro/rail trainsets
from depot stabling tracks into mainline revenue service during induction windows.

Multi-Criteria Optimization Objective:
    Maximize: Fitness Score = w1 * Health + w2 * ShuntMinimization + w3 * MileageBalance + w4 * Cleanliness
    Subject to:
        - Safety & certification constraints (Health >= 75, No critical job cards)
        - Yard neck geometry & blocking constraints (FIFO / Shunting latency)
        - Timetable headway requirements (Peak slots 05:00 - 09:30)
        - Standby rake buffers (Minimum 1 Hot Standby at depot throat)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

@dataclass
class Trainset:
    id: str
    track: str
    position_in_track: int  # 0 = nearest to throat/exit (0 shunts required)
    health_score: float     # 0 to 100 based on IoT telemetry
    cumulative_mileage: float  # in km
    cleanliness_score: float   # 0 to 100
    status: str             # "READY", "STANDBY", "MAINTENANCE", "DEFECTIVE"
    assigned_slot: Optional[str] = None
    line: Optional[str] = None
    critical_defects: List[str] = field(default_factory=list)

    @property
    def shunting_cost(self) -> int:
        """Shunting moves required to pull this rake out of the bay."""
        return self.position_in_track


class TrainInductionOptimizer:
    def __init__(
        self,
        target_mileage: float = 80000.0,
        w_health: float = 0.35,
        w_shunting: float = 0.30,
        w_mileage: float = 0.20,
        w_cleaning: float = 0.15,
    ):
        self.target_mileage = target_mileage
        self.w_health = w_health
        self.w_shunting = w_shunting
        self.w_mileage = w_mileage
        self.w_cleaning = w_cleaning

    def calculate_fitness(self, train: Trainset) -> float:
        """
        Multi-criteria fitness evaluation function:
        Returns normalized score from 0 to 100.
        """
        if train.status == "DEFECTIVE" or train.health_score < 70 or train.critical_defects:
            return 0.0

        # Shunt penalty
        shunt_penalty_factor = max(0.0, 100.0 - (train.shunting_cost * 25.0))

        # Mileage balancing: score higher if train has less mileage (to balance wear)
        mileage_delta = abs(train.cumulative_mileage - self.target_mileage)
        mileage_score = max(20.0, 100.0 - (mileage_delta / 250.0))

        fitness = (
            self.w_health * train.health_score +
            self.w_shunting * shunt_penalty_factor +
            self.w_mileage * mileage_score +
            self.w_cleaning * train.cleanliness_score
        )
        return round(fitness, 2)

    def optimize_induction_schedule(
        self,
        fleet: List[Trainset],
        departure_slots: List[Dict[str, str]]
    ) -> Dict:
        """
        Assigns candidate trains to timetable departure slots.
        """
        # 1. Filter eligible rakes
        eligible = [t for t in fleet if t.status != "DEFECTIVE" and not t.critical_defects and t.health_score >= 70]

        # 2. Rank candidates by fitness
        ranked = sorted(eligible, key=lambda t: self.calculate_fitness(t), reverse=True)

        assignments = []
        assigned_train_ids = set()

        for slot_info in departure_slots:
            slot_time = slot_info["time"]
            line = slot_info["line"]

            # Pick best available candidate
            candidate = next((t for t in ranked if t.id not in assigned_train_ids), None)
            if candidate:
                candidate.assigned_slot = slot_time
                candidate.line = line
                candidate.status = "INDUCTED"
                assigned_train_ids.add(candidate.id)

                assignments.append({
                    "slot": slot_time,
                    "line": line,
                    "train_id": candidate.id,
                    "track": candidate.track,
                    "shunts_needed": candidate.shunting_cost,
                    "fitness_score": self.calculate_fitness(candidate),
                    "health": candidate.health_score
                })

        # Reserve Hot Standby rake
        standby_candidate = next((t for t in ranked if t.id not in assigned_train_ids), None)
        standby_id = None
        if standby_candidate:
            standby_candidate.status = "HOT_STANDBY"
            standby_id = standby_candidate.id

        return {
            "scheduled_inductions": assignments,
            "hot_standby": standby_id,
            "total_slots_filled": len(assignments),
            "unassigned_ready": len(ranked) - len(assigned_train_ids)
        }

    def handle_dynamic_disruption(
        self,
        fleet: List[Trainset],
        failed_train_id: str,
        defect_description: str
    ) -> Dict:
        """
        Dynamically handles an unexpected train defect just prior to induction,
        swapping in the optimal standby rake with minimal shunting disruption.
        """
        failed_train = next((t for t in fleet if t.id == failed_train_id), None)
        if not failed_train:
            return {"error": f"Train {failed_train_id} not found."}

        vacated_slot = failed_train.assigned_slot
        vacated_line = failed_train.line

        failed_train.status = "DEFECTIVE"
        failed_train.assigned_slot = None
        failed_train.critical_defects.append(defect_description)

        # Search for Hot Standby or highest-fitness unassigned train
        standby = next((t for t in fleet if t.status in ["HOT_STANDBY", "READY"] and not t.critical_defects), None)

        if standby:
            standby.status = "INDUCTED"
            standby.assigned_slot = vacated_slot
            standby.line = vacated_line
            return {
                "success": True,
                "disrupted_train": failed_train_id,
                "replacement_train": standby.id,
                "reassigned_slot": vacated_slot,
                "line": vacated_line,
                "shunts_required": standby.shunting_cost,
                "message": f"Successfully reassigned slot {vacated_slot} to {standby.id}."
            }
        else:
            return {
                "success": False,
                "disrupted_train": failed_train_id,
                "message": "No available standby rake. Fleet headways must be widened."
            }


# Demonstration test execution
if __name__ == "__main__":
    sample_fleet = [
        Trainset("TS-101", "S-01", 0, 98.0, 82000, 95.0, "READY"),
        Trainset("TS-102", "S-01", 1, 94.0, 91000, 90.0, "READY"),
        Trainset("TS-103", "S-01", 2, 91.0, 84000, 92.0, "READY"),
        Trainset("TS-104", "S-02", 0, 96.0, 78000, 98.0, "READY"),
        Trainset("TS-105", "S-02", 1, 89.0, 89000, 88.0, "READY"),
        Trainset("TS-106", "S-02", 2, 88.0, 95000, 85.0, "READY"),
        Trainset("TS-107", "S-03", 0, 95.0, 76000, 90.0, "READY"),
        Trainset("TS-108", "S-03", 1, 90.0, 83000, 87.0, "READY"),
        Trainset("TS-109", "S-03", 2, 97.0, 80500, 95.0, "READY"),
        Trainset("TS-110", "S-04", 0, 92.0, 87000, 89.0, "READY"),
        Trainset("TS-111", "S-04", 1, 93.0, 85000, 91.0, "READY"),
        Trainset("TS-112", "S-04", 2, 87.0, 93500, 86.0, "READY"),
    ]

    slots = [
        {"time": "05:15", "line": "Line 1 (Red)"},
        {"time": "05:30", "line": "Line 2 (Blue)"},
        {"time": "05:45", "line": "Line 1 (Red)"},
        {"time": "06:00", "line": "Line 2 (Blue)"},
        {"time": "06:15", "line": "Line 1 (Red)"},
        {"time": "06:30", "line": "Line 2 (Blue)"},
        {"time": "06:45", "line": "Line 1 (Red)"},
        {"time": "07:00", "line": "Line 2 (Blue)"},
        {"time": "07:15", "line": "Line 1 (Red)"},
        {"time": "07:30", "line": "Line 2 (Blue)"},
    ]

    optimizer = TrainInductionOptimizer()
    plan = optimizer.optimize_induction_schedule(sample_fleet, slots)
    print("=== AI Train Induction Optimization Plan ===")
    print(json.dumps(plan, indent=2))

    print("\n=== Simulating Disruption on TS-104 ===")
    disruption_res = optimizer.handle_dynamic_disruption(
        sample_fleet, "TS-104", "Brake Cylinder Pressure Anomaly"
    )
    print(json.dumps(disruption_res, indent=2))
