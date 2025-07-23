# workforce_simulator.py
#
# This script simulates workforce demand over time based on a pipeline of construction projects.
# It uses predefined rules for project duration and staffing requirements to forecast
# the number of employees needed for each role on a month-by-month basis.
#
# NEW: Fixed the date calculation logic to correctly advance by one calendar month.

import math
import random
import pandas as pd
from datetime import datetime

# --- PHASE 1: DEFINING THE CORE RULES & COEFFICIENTS ---
# These dictionaries store the business logic you provided.

# Rule 1: Project Durations in months
PROJECT_DURATIONS = {
    "Pre-analysis": { "Small": 3, "Medium": 6, "Large": 8 },
    "Actual Work": { "Small": 11, "Medium": 20, "Large": 25 }
}

# Rule 2: Staffing Coefficients (Fraction of one employee required per project)
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

# --- PHASE 2: PROJECT AND SIMULATION SETUP ---

class Project:
    """A class to represent a single construction project and track its progress."""
    def __init__(self, id, p_type, p_scale, start_month):
        self.id = id
        self.p_type = p_type
        self.p_scale = p_scale
        self.start_month = start_month
        self.duration = PROJECT_DURATIONS[p_type][p_scale]
        self.end_month = start_month + self.duration - 1

    def is_active_in_month(self, month):
        return self.start_month <= month <= self.end_month

    def get_staffing_needs(self):
        return STAFFING_COEFFICIENTS.get(self.p_type, {})

def run_simulation(all_years_pa_counts, all_years_aw_base_counts, conversion_rate, simulation_duration_months, start_date_str='2026-01-01'):
    """
    Runs the main workforce simulation and returns a DataFrame with the results.
    """
    print("--- Starting Workforce Simulation ---")
    print(f"Simulating for {simulation_duration_months} months.")
    print(f"Conversion Rate: {conversion_rate * 100}%\n")

    project_pipeline = []
    project_id_counter = 0
    results_data = []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    scales = ["Small", "Medium", "Large"]

    # --- PHASE 3: MONTH-BY-MONTH CALCULATION & DYNAMIC GENERATION ---
    for month in range(1, simulation_duration_months + 1):
        
        # --- Part A: Generate new projects for this month ---
        current_year_index = (month - 1) // 12
        if current_year_index < len(all_years_pa_counts):
            yearly_pa_counts = all_years_pa_counts[current_year_index]
            yearly_aw_base_counts = all_years_aw_base_counts[current_year_index]
            month_within_year = (month - 1) % 12
            
            for p_scale, yearly_count in yearly_pa_counts.items():
                num_to_add = yearly_count // 12 + (1 if month_within_year < yearly_count % 12 else 0)
                for _ in range(num_to_add):
                    project_id_counter += 1
                    project_pipeline.append(Project(f"PA_{p_scale}_{project_id_counter}", "Pre-analysis", p_scale, month))

            for p_scale, yearly_count in yearly_aw_base_counts.items():
                num_to_add = yearly_count // 12 + (1 if month_within_year < yearly_count % 12 else 0)
                for _ in range(num_to_add):
                    project_id_counter += 1
                    project_pipeline.append(Project(f"AW-base_{p_scale}_{project_id_counter}", "Actual Work", p_scale, month))

        # --- Part B: Handle conversions ---
        converted_this_month = 0
        finished_pa_projects = [p for p in project_pipeline if p.p_type == "Pre-analysis" and p.end_month == month - 1]
        for proj in finished_pa_projects:
            if random.random() < conversion_rate:
                converted_this_month += 1
                project_id_counter += 1
                project_pipeline.append(Project(f"AW-Conv_{proj.p_scale}_{project_id_counter}", "Actual Work", p_scale, month))

        # --- Part C: Calculate staffing and active project counts ---
        monthly_demand = {role: {"Small": 0, "Medium": 0, "Large": 0} for role in roles}
        active_project_counts = {"PA Small": 0, "PA Medium": 0, "PA Large": 0, "AW Small": 0, "AW Medium": 0, "AW Large": 0}

        for proj in project_pipeline:
            if proj.is_active_in_month(month):
                key = f"{'PA' if proj.p_type == 'Pre-analysis' else 'AW'} {proj.p_scale}"
                active_project_counts[key] += 1
                
                project_needs = proj.get_staffing_needs()
                for role, requirements_by_scale in project_needs.items():
                    if proj.p_scale in requirements_by_scale:
                        demand_value = requirements_by_scale[proj.p_scale]
                        monthly_demand[role][proj.p_scale] += demand_value
        
        # --- Part D: Store results for this month ---
        # FIXED: Use pandas DateOffset for accurate month progression.
        current_date = start_date + pd.DateOffset(months=month - 1)
        month_data = { "Date": current_date.strftime('%Y-%m'), "Converted Projects": converted_this_month, **active_project_counts }

        for role in roles:
            total_hired_for_role = 0
            for scale in scales:
                demand = monthly_demand[role][scale]
                hired_count = math.ceil(demand)
                month_data[f"{role} Hired ({scale})"] = hired_count
                total_hired_for_role += hired_count
            month_data[f"{role} Hired (Total)"] = total_hired_for_role
        
        results_data.append(month_data)

    # --- PHASE 4: CREATE AND RETURN DATAFRAME ---
    results_df = pd.DataFrame(results_data)
    
    column_order = ["Date", "Converted Projects", "PA Small", "PA Medium", "PA Large", "AW Small", "AW Medium", "AW Large"]
    for role in roles:
        for scale in scales:
            column_order.append(f"{role} Hired ({scale})")
        column_order.append(f"{role} Hired (Total)")

    results_df = results_df.reindex(columns=column_order, fill_value=0)
    
    # Set Date as the index for easier plotting
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    # results_df = results_df.set_index('Date')
    
    for col in results_df.columns:
        if 'Hired' in col or 'Projects' in col:
            results_df[col] = results_df[col].astype(int)

    print("--- Simulation Complete. DataFrame generated. ---")
    return results_df


# --- EXAMPLE USAGE (for a .py file or a notebook cell) ---
if __name__ == "__main__":
    ALL_YEARS_PA_PROJECTS = [
        {"Small": 44, "Medium": 64, "Large": 12},  # Year 1
        {"Small": 26, "Medium": 38, "Large": 7}    # Year 2
    ]
    ALL_YEARS_AW_BASE_PROJECTS = [
        {"Small": 44, "Medium": 5, "Large": 10},   # Year 1
        {"Small": 20, "Medium": 3, "Large": 12}    # Year 2
    ]
    
    CONVERSION_RATE = 1

    # In your notebook, you would run this cell to get the DataFrame
    simulation_df = run_simulation(
        all_years_pa_counts=ALL_YEARS_PA_PROJECTS,
        all_years_aw_base_counts=ALL_YEARS_AW_BASE_PROJECTS,
        conversion_rate=CONVERSION_RATE,
        simulation_duration_months=60,
        start_date_str='2026-01-01' # You can change the start date here
    )

    # Now you can work with the DataFrame in subsequent cells.
    # For example, display the first few rows:
    print("\n--- Simulation Results DataFrame ---")
    # Reset index to show the 'Date' column for verification
    print(simulation_df.head(12).reset_index()[['Date']])
