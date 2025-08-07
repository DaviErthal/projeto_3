# Rule 1: Default Project Durations in months
PROJECT_DURATIONS = {
    "Pre-analysis": { "Small": 3, "Medium": 6, "Large": 8 },
    "Actual Work": { "Small": 11, "Medium": 20, "Large": 25 }
}

# Rule 2: Default Staffing Coefficients
STAFFING_COEFFICIENTS = {
    "Pre-analysis": {
        "Analyst": {"Small": 1/20, "Medium": 1/12, "Large": 1/6},
        "Technical": {"Small": 1/20, "Medium": 1/12, "Large": 1/6},
    },
    "Actual Work": {
        "Project Manager": {"Small": 0, "Medium": 1/6, "Large": 1/6},
        "Tax Engineer": {"Small": 1/12, "Medium": 1/6, "Large": 1/3},
        "Tax Technical": {"Small": 1/5, "Medium": 1/5, "Large": 1.0},
        "Planning Engineer": {"Small": 0, "Medium": 1/12, "Large": 1/6},
    }
}

# Default Monthly Salaries
DEFAULT_SALARIES = {
    "Analyst": 16500, "Technical": 3100, "Project Manager": 18000,
    "Tax Engineer": 16500, "Tax Technical": 3100, "Planning Engineer": 16500
}
# --- ADDED: Default per-employee overhead cost ---
DEFAULT_MEAN_EMPLOYEE_COST = 10


# Default Monthly Revenues
DEFAULT_REVENUES = {
    "Pre-analysis": {"Small": 4600, "Medium": 6250, "Large": 11828},
    "Actual Work": {"Small": 10700, "Medium": 39200, "Large": 68600}
}

# --- Data Scenarios ---
SCENARIOS = {
    "Lote 1": {
        "pa": [{"Small": 44, "Medium": 64, "Large": 12}, {"Small": 26, "Medium": 38, "Large": 7}],
        "aw": [{"Small": 44, "Medium": 5, "Large": 10}, {"Small": 20, "Medium": 3, "Large": 12}]
    },
    "Lote 2": {
        "pa": [{"Small": 36, "Medium": 40, "Large": 16}, {"Small": 21, "Medium": 24, "Large": 9}],
        "aw": [{"Small": 37, "Medium": 14, "Large": 16}, {"Small": 33, "Medium": 9, "Large": 9}]
    },
    "Lote 3": {
        "pa": [{"Small": 36, "Medium": 72, "Large": 16}, {"Small": 21, "Medium": 42, "Large": 9}],
        "aw": [{"Small": 5, "Medium": 2, "Large": 12}, {"Small": 7, "Medium": 1, "Large": 81}]
    },
    "Personalized": {}
}
