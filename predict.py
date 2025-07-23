# workforce_simulator.py
#
# This script simulates workforce demand over time based on a pipeline of construction projects.
# It uses predefined rules for project duration and staffing requirements to forecast
# the number of employees needed for each role on a month-by-month basis.
#
# NEW: The output is now a pandas DataFrame for structured analysis.

import math
import random
import pandas as pd
from datetime import datetime, timedelta

# --- PHASE 1: DEFINING THE CORE RULES & COEFFICIENTS ---
# These dictionaries store the business logic you provided.

# Rule 1: Project Durations in months
PROJECT_DURATIONS = {
    "Pre-analysis": {
        "Small": 6,
        "Medium": 6,
        "Large": 6,
    },
    "Actual Work": {
        "Small": 6,
        "Medium": 12,
        "Large": 24,
    }
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
        """
        Initializes a project.
        
        Args:
            id (str): A unique identifier for the project.
            p_type (str): "Pre-analysis" or "Actual Work".
            p_scale (str): "Small", "Medium", or "Large".
            start_month (int): The month number when the project begins (e.g., 1 for January).
        """
        self.id = id
        self.p_type = p_type
        self.p_scale = p_scale
        self.start_month = start_month
        self.duration = PROJECT_DURATIONS[p_type][p_scale]
        self.end_month = start_month + self.duration - 1

    def is_active_in_month(self, month):
        """Checks if the project is active during a given month."""
        return self.start_month <= month <= self.end_month

    def get_staffing_needs(self):
        """Returns the dictionary of staff required for this project."""
        return STAFFING_COEFFICIENTS.get(self.p_type, {})

def run_simulation(all_years_pa_counts, all_years_aw_base_counts, conversion_rate, simulation_duration_months):
    """
    Runs the main workforce simulation, generating a DataFrame with the results.
    
    Args:
        all_years_pa_counts (list of dict): Yearly totals for new Pre-analysis projects.
        all_years_aw_base_counts (list of dict): Yearly totals for new "base" Actual Work projects.
        conversion_rate (float): The probability (0.0 to 1.0) that a PA project converts to AW.
        simulation_duration_months (int): The total number of months to simulate.
    """
    print("--- Starting Workforce Simulation ---")
    print(f"Simulating for {simulation_duration_months} months.")
    print(f"Conversion Rate: {conversion_rate * 100}%\n")

    project_pipeline = []
    project_id_counter = 0
    
    # List to hold the data for each month before creating the DataFrame
    results_data = []
    start_date = datetime(2025, 1, 1) # Set a base start date for reporting

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
        monthly_demand = {"Analyst": 0, "Technical": 0, "Project Manager": 0, "Tax Engineer": 0, "Tax Technical": 0, "Planning Engineer": 0}
        active_project_counts = {"PA Small": 0, "PA Medium": 0, "PA Large": 0, "AW Small": 0, "AW Medium": 0, "AW Large": 0}

        for proj in project_pipeline:
            if proj.is_active_in_month(month):
                # Increment active project count
                key = f"{'PA' if proj.p_type == 'Pre-analysis' else 'AW'} {proj.p_scale}"
                active_project_counts[key] += 1
                
                # Add to staffing demand
                project_needs = proj.get_staffing_needs()
                for role, requirement in project_needs.items():
                    if proj.p_scale in requirement:
                        monthly_demand[role] += requirement[proj.p_scale]
        
        # --- Part D: Store results for this month ---
        current_date = start_date + timedelta(days=30 * (month - 1))
        month_data = {
            "Date": current_date.strftime('%Y-%m'),
            "Converted Projects": converted_this_month,
            **active_project_counts, # Unpack the project counts into columns
            **{f"{role} Demand": round(demand, 2) for role, demand in monthly_demand.items()}, # Add rounded staffing demand
            **{f"{role} Hired": math.ceil(demand) for role, demand in monthly_demand.items()} # Add ceiling for hired staff
        }
        results_data.append(month_data)

    # --- PHASE 4: CREATE AND DISPLAY DATAFRAME ---
    results_df = pd.DataFrame(results_data)
    
    # Reorder columns for better readability
    column_order = [
        "Date", "Converted Projects", "PA Small", "PA Medium", "PA Large", "AW Small", "AW Medium", "AW Large",
        "Analyst Hired", "Technical Hired", "Project Manager Hired", "Tax Engineer Hired", "Tax Technical Hired", "Planning Engineer Hired"
    ]
    # Add the demand columns as well, if you want to see the fractional values
    # column_order += [col for col in results_df.columns if "Demand" in col]

    results_df = results_df[column_order]

    print("--- Simulation Results ---")
    # Use to_string() to ensure all columns are displayed in the console
    print(results_df.to_string())


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Define the new inputs based on your request for two years.
    # This is now a list of dictionaries, where each dictionary represents one year.
    ALL_YEARS_PA_PROJECTS = [
        {"Small": 44, "Medium": 64, "Large": 12},  # Year 1
        {"Small": 26, "Medium": 38, "Large": 7}    # Year 2
    ]
    ALL_YEARS_AW_BASE_PROJECTS = [
        {"Small": 44, "Medium": 5, "Large": 10},   # Year 1
        {"Small": 20, "Medium": 3, "Large": 12}    # Year 2
    ]
    
    # Set the conversion rate 'c'. 0.5 means 50% of Pre-analysis projects
    # will become Actual Work projects the month after they finish.
    CONVERSION_RATE = 0.5 

    # Run the simulation for 36 months to see the long-term effects.
    run_simulation(
        all_years_pa_counts=ALL_YEARS_PA_PROJECTS,
        all_years_aw_base_counts=ALL_YEARS_AW_BASE_PROJECTS,
        conversion_rate=CONVERSION_RATE,
        simulation_duration_months=36
    )
