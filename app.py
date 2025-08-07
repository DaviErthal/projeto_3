# app.py
#
# This script creates an interactive web-based dashboard for the workforce simulation
# using the Dash framework by Plotly.
#
# NEW:
# 1. Added Monte Carlo simulation for risk analysis on Project Durations.
# 2. Added a new confidence interval graph for cumulative cash flow.

import math
import random
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# You will need to install dash, pandas, plotly, and openpyxl
# pip install dash pandas plotly openpyxl numpy
import dash
from dash import dcc, html, ALL
from dash.dependencies import Input, Output, State

# --- PHASE 1: SIMULATION LOGIC ---

# Default Project Durations in months
PROJECT_DURATIONS = {
    "Pre-analysis": { "Small": 3, "Medium": 6, "Large": 8 },
    "Actual Work": { "Small": 11, "Medium": 20, "Large": 25 }
}

# Default Staffing Coefficients
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
DEFAULT_MEAN_EMPLOYEE_COST = 500

# Default Monthly Revenues
DEFAULT_REVENUES = {
    "Pre-analysis": {"Small": 4600, "Medium": 6250, "Large": 11828},
    "Actual Work": {"Small": 10700, "Medium": 39200, "Large": 68600}
}

# --- Data Scenarios (Hardcoded for 2 years) ---
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


class Project:
    def __init__(self, id, p_type, p_scale, start_month, project_durations, staffing_coefficients):
        self.id, self.p_type, self.p_scale, self.start_month = id, p_type, p_scale, start_month
        self.duration = project_durations[p_type][p_scale]
        self.staffing_coefficients = staffing_coefficients
        self.end_month = start_month + self.duration - 1
    def is_active_in_month(self, month):
        return self.start_month <= month <= self.end_month
    def get_staffing_needs(self):
        return self.staffing_coefficients.get(self.p_type, {})

def run_simulation(all_years_pa_counts, all_years_aw_base_counts, conversion_rate, simulation_duration_months, project_durations, staffing_coefficients, start_date_str='2026-01-01'):
    project_pipeline, project_id_counter, results_data = [], 0, []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    scales = ["Small", "Medium", "Large"]

    for month in range(1, simulation_duration_months + 1):
        current_year_index = (month - 1) // 12
        if current_year_index < len(all_years_pa_counts):
            yearly_pa_counts = all_years_pa_counts[current_year_index]
            yearly_aw_base_counts = all_years_aw_base_counts[current_year_index]
            month_within_year = (month - 1) % 12
            for p_scale, yearly_count in yearly_pa_counts.items():
                num_to_add = yearly_count // 12 + (1 if month_within_year < yearly_count % 12 else 0)
                for _ in range(num_to_add):
                    project_id_counter += 1
                    project_pipeline.append(Project(f"PA_{p_scale}_{project_id_counter}", "Pre-analysis", p_scale, month, project_durations, staffing_coefficients))
            for p_scale, yearly_count in yearly_aw_base_counts.items():
                num_to_add = yearly_count // 12 + (1 if month_within_year < yearly_count % 12 else 0)
                for _ in range(num_to_add):
                    project_id_counter += 1
                    project_pipeline.append(Project(f"AW-base_{p_scale}_{project_id_counter}", "Actual Work", p_scale, month, project_durations, staffing_coefficients))

        converted_this_month = 0
        finished_pa_projects = [p for p in project_pipeline if p.p_type == "Pre-analysis" and p.end_month == month - 1]
        for proj in finished_pa_projects:
            if random.random() < conversion_rate:
                converted_this_month += 1
                project_id_counter += 1
                project_pipeline.append(Project(f"AW-Conv_{proj.p_scale}_{project_id_counter}", "Actual Work", proj.p_scale, month, project_durations, staffing_coefficients))

        monthly_demand = {role: {"Small": 0, "Medium": 0, "Large": 0} for role in roles}
        active_project_counts = {"PA Small": 0, "PA Medium": 0, "PA Large": 0, "AW Small": 0, "AW Medium": 0, "AW Large": 0}
        for proj in project_pipeline:
            if proj.is_active_in_month(month):
                key = f"{'PA' if proj.p_type == 'Pre-analysis' else 'AW'} {proj.p_scale}"
                active_project_counts[key] += 1
                project_needs = proj.get_staffing_needs()
                for role, requirements_by_scale in project_needs.items():
                    if proj.p_scale in requirements_by_scale:
                        monthly_demand[role][proj.p_scale] += requirements_by_scale[proj.p_scale]
        
        current_date = start_date + pd.DateOffset(months=month - 1)
        month_data = { "Date": current_date.strftime('%Y-%m'), "Converted Projects": converted_this_month, **active_project_counts }
        for role in roles:
            total_hired_for_role = 0
            for scale in scales:
                hired_count = math.ceil(monthly_demand[role][scale])
                month_data[f"{role} Hired ({scale})"] = hired_count
                total_hired_for_role += hired_count
            month_data[f"{role} Hired (Total)"] = total_hired_for_role
        results_data.append(month_data)

    results_df = pd.DataFrame(results_data)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    return results_df

# --- PHASE 2: DASH APPLICATION SETUP ---

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Helper functions to create UI components ---
def create_year_inputs(year):
    return html.Div([
        html.H4(f"Year {year}", style={'marginTop': '20px'}),
        html.Div([
            html.Div([
                html.Label("PA Small:"), dcc.Input(id={'type': 'pa-small-input', 'index': year}, type='number', value=0, style={'width': '80px'}),
                html.Label("PA Medium:"), dcc.Input(id={'type': 'pa-medium-input', 'index': year}, type='number', value=0, style={'width': '80px'}),
                html.Label("PA Large:"), dcc.Input(id={'type': 'pa-large-input', 'index': year}, type='number', value=0, style={'width': '80px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
            html.Div([
                html.Label("AW Small:"), dcc.Input(id={'type': 'aw-small-input', 'index': year}, type='number', value=0, style={'width': '80px'}),
                html.Label("AW Medium:"), dcc.Input(id={'type': 'aw-medium-input', 'index': year}, type='number', value=0, style={'width': '80px'}),
                html.Label("AW Large:"), dcc.Input(id={'type': 'aw-large-input', 'index': year}, type='number', value=0, style={'width': '80px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        ])
    ])

def create_coefficient_inputs():
    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    inputs = []
    for role in roles:
        role_inputs = [html.H5(role, style={'textAlign': 'center', 'marginTop': '15px'})]
        sanitized_role = role.replace(" ", "").lower()
        pa_div = [html.Label("Pre-analysis: ")]
        for scale in ["Small", "Medium", "Large"]:
            pa_div.append(html.Label(f"{scale}:"))
            pa_div.append(dcc.Input(id=f'coef-{sanitized_role}-pa-{scale.lower()}', type='number', value=round(STAFFING_COEFFICIENTS.get("Pre-analysis", {}).get(role, {}).get(scale, 0), 3), step=0.001, style={'width': '80px', 'marginRight': '10px'}))
        role_inputs.append(html.Div(pa_div))
        aw_div = [html.Label("Actual Work: ")]
        for scale in ["Small", "Medium", "Large"]:
            aw_div.append(html.Label(f"{scale}:"))
            aw_div.append(dcc.Input(id=f'coef-{sanitized_role}-aw-{scale.lower()}', type='number', value=round(STAFFING_COEFFICIENTS.get("Actual Work", {}).get(role, {}).get(scale, 0), 3), step=0.001, style={'width': '80px', 'marginRight': '10px'}))
        role_inputs.append(html.Div(aw_div))
        inputs.append(html.Div(role_inputs, style={'borderTop': '1px solid #eee', 'paddingTop': '10px'}))
    return html.Div(inputs)

def create_cost_inputs():
    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    inputs = [
        html.Div([
            html.Label("Mean Employee Overhead Cost:"),
            dcc.Input(id='mean-employee-cost', type='number', value=DEFAULT_MEAN_EMPLOYEE_COST, style={'width': '100px'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'})
    ]
    salary_inputs = []
    for role in roles:
        sanitized_role = role.replace(" ", "").lower()
        salary_inputs.append(html.Label(f"{role} Salary:"))
        salary_inputs.append(dcc.Input(id=f'salary-{sanitized_role}', type='number', value=DEFAULT_SALARIES.get(role, 0), style={'width': '100px', 'marginRight': '20px'}))
    inputs.append(html.Div(salary_inputs, style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}))
    return html.Div(inputs)

def create_revenue_inputs():
    inputs = []
    for p_type in ["Pre-analysis", "Actual Work"]:
        type_inputs = [html.H5(p_type, style={'textAlign': 'center', 'marginTop': '15px'})]
        div_children = []
        for scale in ["Small", "Medium", "Large"]:
            sanitized_type = p_type.replace("-", "").replace(" ", "").lower()
            div_children.append(html.Label(f"{scale} Revenue:"))
            div_children.append(dcc.Input(id=f'revenue-{sanitized_type}-{scale.lower()}', type='number', value=DEFAULT_REVENUES[p_type][scale], style={'width': '100px', 'marginRight': '20px'}))
        type_inputs.append(html.Div(div_children, style={'display': 'flex', 'justifyContent': 'space-around'}))
        inputs.append(html.Div(type_inputs))
    return html.Div(inputs)

def create_recurring_entry_inputs():
    rows = []
    for i in range(1, 4):
        rows.append(html.Div([
            dcc.Input(id={'type': 'entry-name', 'index': i}, type='text', placeholder=f'Entry #{i} Name', style={'width': '25%'}),
            dcc.Input(id={'type': 'entry-start', 'index': i}, type='text', placeholder='Start (YYYY-MM)', style={'width': '20%'}),
            dcc.Input(id={'type': 'entry-end', 'index': i}, type='text', placeholder='End (YYYY-MM)', style={'width': '20%'}),
            dcc.Input(id={'type': 'entry-value', 'index': i}, type='number', placeholder='Monthly Value', style={'width': '15%'}),
            dcc.RadioItems(
                id={'type': 'entry-type', 'index': i},
                options=[{'label': 'Expense', 'value': 'expense'}, {'label': 'Revenue', 'value': 'revenue'}],
                value='expense',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px', 'alignItems': 'center'}))
    return html.Div(rows)


# Define the layout of the application
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Workforce Planning & Optimization Dashboard", style={'textAlign': 'center', 'color': '#333'}),
    html.P("Use the controls below to run different scenarios and forecast staffing needs.", style={'textAlign': 'center', 'color': '#666'}),

    # Control Panel
    html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-around'}, children=[
            html.Div([
                html.Label("Select Project Scenario:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(id='scenario-dropdown', options=[{'label': key, 'value': key} for key in SCENARIOS.keys()], value='Lote 1', clearable=False)
            ], style={'width': '30%'}),
            html.Div([
                html.Label("Conversion Rate (%):", style={'fontWeight': 'bold'}),
                dcc.Slider(id='conversion-rate-slider', min=0, max=100, step=5, value=50, marks={i: f'{i}%' for i in range(0, 101, 10)})
            ], style={'width': '30%'}),
            html.Div([
                html.Label("Simulation Duration (Months):", style={'fontWeight': 'bold'}),
                dcc.Input(id='duration-input', type='number', value=36, min=12, step=1)
            ], style={'width': '20%'})
        ]),
        
        html.Div(id='personalized-inputs-container', style={'display': 'none', 'marginTop': '20px', 'borderTop': '1px solid #ccc', 'paddingTop': '20px'}, children=[
            html.H3("Enter Personalized Scenario Data", style={'textAlign': 'center'}),
            html.Div([
                html.Label("Number of Years to Plan:"),
                dcc.Input(id='num-years-input', type='number', value=2, min=1, max=10, step=1)
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div(id='dynamic-year-inputs-container') # Container for dynamic year inputs
        ]),
        
        html.Details([
            html.Summary('Advanced Settings'),
            html.Div([
                html.Details([
                    html.Summary('Customize Durations'),
                    html.Div(style={'marginTop': '10px'}, children=[
                        html.Div([
                            html.Label("PA Small:"), dcc.Input(id='dur-pa-small', type='number', value=PROJECT_DURATIONS["Pre-analysis"]["Small"], style={'width': '80px'}),
                            html.Label("PA Medium:"), dcc.Input(id='dur-pa-medium', type='number', value=PROJECT_DURATIONS["Pre-analysis"]["Medium"], style={'width': '80px'}),
                            html.Label("PA Large:"), dcc.Input(id='dur-pa-large', type='number', value=PROJECT_DURATIONS["Pre-analysis"]["Large"], style={'width': '80px'}),
                        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
                        html.Div([
                            html.Label("AW Small:"), dcc.Input(id='dur-aw-small', type='number', value=PROJECT_DURATIONS["Actual Work"]["Small"], style={'width': '80px'}),
                            html.Label("AW Medium:"), dcc.Input(id='dur-aw-medium', type='number', value=PROJECT_DURATIONS["Actual Work"]["Medium"], style={'width': '80px'}),
                            html.Label("AW Large:"), dcc.Input(id='dur-aw-large', type='number', value=PROJECT_DURATIONS["Actual Work"]["Large"], style={'width': '80px'}),
                        ], style={'display': 'flex', 'justifyContent': 'space-around'}),
                    ])
                ]),
                html.Details([
                    html.Summary('Customize Staffing Coefficients'),
                    html.Div(create_coefficient_inputs(), style={'marginTop': '10px'})
                ]),
                html.Details([
                    html.Summary('Customize Costs'),
                    html.Div(create_cost_inputs(), style={'marginTop': '10px'})
                ]),
                html.Details([
                    html.Summary('Customize Revenues (Per Project/Month)'),
                    html.Div(create_revenue_inputs(), style={'marginTop': '10px'})
                ]),
                html.Details([
                    html.Summary('Add Recurring Entries (Revenues/Expenses)'),
                    html.Div(create_recurring_entry_inputs(), style={'marginTop': '10px'})
                ])
            ])
        ], style={'marginTop': '20px'}),
        
        # --- ADDED: Risk Analysis Section ---
        html.Div(style={'borderTop': '1px solid #ccc', 'marginTop': '20px', 'paddingTop': '20px'}, children=[
            html.H3("Risk Analysis (Monte Carlo)", style={'textAlign': 'center'}),
            html.Div([
                html.Label("Number of Simulations to Run:"),
                dcc.Input(id='mc-runs-input', type='number', value=100, min=1, step=10, style={'width': '100px', 'marginLeft': '10px'})
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("This will run the simulation multiple times with random variations in Project Durations and Employee Overhead to see a range of possible outcomes.", style={'textAlign': 'center', 'fontSize': '14px', 'color': '#666'})
        ])
    ]),

    html.Div(style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
        html.Button('Run Simulation', id='run-button', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px'})
    ]),
    
    dcc.Loading(id="loading-icon", children=[html.Div(id='output-graphs')], type="circle"),
])

# --- PHASE 3: CALLBACKS TO CONNECT UI TO SIMULATION ---

@app.callback(
    Output('personalized-inputs-container', 'style'),
    Input('scenario-dropdown', 'value')
)
def toggle_personalized_inputs(scenario_name):
    return {'display': 'block', 'marginTop': '20px', 'borderTop': '1px solid #ccc', 'paddingTop': '20px'} if scenario_name == 'Personalized' else {'display': 'none'}

@app.callback(
    Output('dynamic-year-inputs-container', 'children'),
    Input('num-years-input', 'value')
)
def update_year_inputs(num_years):
    if num_years is None or num_years < 1:
        return []
    return [create_year_inputs(i) for i in range(1, num_years + 1)]


@app.callback(
    Output('output-graphs', 'children'),
    Input('run-button', 'n_clicks'),
    [
        State('scenario-dropdown', 'value'), State('conversion-rate-slider', 'value'), State('duration-input', 'value'),
        # Dynamic year inputs
        State({'type': 'pa-small-input', 'index': ALL}, 'value'), State({'type': 'pa-medium-input', 'index': ALL}, 'value'), State({'type': 'pa-large-input', 'index': ALL}, 'value'),
        State({'type': 'aw-small-input', 'index': ALL}, 'value'), State({'type': 'aw-medium-input', 'index': ALL}, 'value'), State({'type': 'aw-large-input', 'index': ALL}, 'value'),
        # Custom duration inputs
        State('dur-pa-small', 'value'), State('dur-pa-medium', 'value'), State('dur-pa-large', 'value'),
        State('dur-aw-small', 'value'), State('dur-aw-medium', 'value'), State('dur-aw-large', 'value'),
        # Custom coefficient inputs
        State('coef-analyst-pa-small', 'value'), State('coef-analyst-pa-medium', 'value'), State('coef-analyst-pa-large', 'value'),
        State('coef-analyst-aw-small', 'value'), State('coef-analyst-aw-medium', 'value'), State('coef-analyst-aw-large', 'value'),
        State('coef-technical-pa-small', 'value'), State('coef-technical-pa-medium', 'value'), State('coef-technical-pa-large', 'value'),
        State('coef-technical-aw-small', 'value'), State('coef-technical-aw-medium', 'value'), State('coef-technical-aw-large', 'value'),
        State('coef-projectmanager-pa-small', 'value'), State('coef-projectmanager-pa-medium', 'value'), State('coef-projectmanager-pa-large', 'value'),
        State('coef-projectmanager-aw-small', 'value'), State('coef-projectmanager-aw-medium', 'value'), State('coef-projectmanager-aw-large', 'value'),
        State('coef-taxengineer-pa-small', 'value'), State('coef-taxengineer-pa-medium', 'value'), State('coef-taxengineer-pa-large', 'value'),
        State('coef-taxengineer-aw-small', 'value'), State('coef-taxengineer-aw-medium', 'value'), State('coef-taxengineer-aw-large', 'value'),
        State('coef-taxtechnical-pa-small', 'value'), State('coef-taxtechnical-pa-medium', 'value'), State('coef-taxtechnical-pa-large', 'value'),
        State('coef-taxtechnical-aw-small', 'value'), State('coef-taxtechnical-aw-medium', 'value'), State('coef-taxtechnical-aw-large', 'value'),
        State('coef-planningengineer-pa-small', 'value'), State('coef-planningengineer-pa-medium', 'value'), State('coef-planningengineer-pa-large', 'value'),
        State('coef-planningengineer-aw-small', 'value'), State('coef-planningengineer-aw-medium', 'value'), State('coef-planningengineer-aw-large', 'value'),
        # Salary and overhead inputs
        State('salary-analyst', 'value'), State('salary-technical', 'value'), State('salary-projectmanager', 'value'),
        State('salary-taxengineer', 'value'), State('salary-taxtechnical', 'value'), State('salary-planningengineer', 'value'),
        State('mean-employee-cost', 'value'),
        # Revenue inputs
        State('revenue-preanalysis-small', 'value'), State('revenue-preanalysis-medium', 'value'), State('revenue-preanalysis-large', 'value'),
        State('revenue-actualwork-small', 'value'), State('revenue-actualwork-medium', 'value'), State('revenue-actualwork-large', 'value'),
        # Recurring entry inputs
        State({'type': 'entry-name', 'index': ALL}, 'value'), State({'type': 'entry-start', 'index': ALL}, 'value'),
        State({'type': 'entry-end', 'index': ALL}, 'value'), State({'type': 'entry-value', 'index': ALL}, 'value'),
        State({'type': 'entry-type', 'index': ALL}, 'value'),
        # --- ADDED: State for Monte Carlo runs ---
        State('mc-runs-input', 'value'),
    ]
)
def update_dashboard(n_clicks, scenario_name, conversion_rate_percent, duration,
                     pa_s_all, pa_m_all, pa_l_all, aw_s_all, aw_m_all, aw_l_all,
                     dur_pa_s, dur_pa_m, dur_pa_l, dur_aw_s, dur_aw_m, dur_aw_l,
                     *args): 
    if n_clicks == 0:
        return html.Div("Please set your parameters and click 'Run Simulation'.", style={'textAlign': 'center', 'padding': '50px', 'fontSize': '18px'})

    # Unpack all arguments from *args
    coeffs = args[:36]
    salaries_tuple = args[36:42]
    mean_employee_cost_base = args[42]
    revenues_tuple = args[43:49]
    entry_names, entry_starts, entry_ends, entry_values, entry_types = args[49:54]
    mc_runs = args[54]

    # --- FIX: Ensure mc_runs is at least 1 to prevent crash ---
    if mc_runs is None or mc_runs < 1:
        mc_runs = 1

    # --- MONTE CARLO SIMULATION LOOP ---
    all_runs_cash_flow = []
    # Store the last run's df for the representative graphs
    last_run_df = None
    
    for i in range(mc_runs):
        # --- Introduce randomness for this specific run ---
        # Randomize durations by +/- 15%
        randomized_durations = {
            "Pre-analysis": {
                "Small": max(1, round(dur_pa_s * random.uniform(0.85, 1.15))),
                "Medium": max(1, round(dur_pa_m * random.uniform(0.85, 1.15))),
                "Large": max(1, round(dur_pa_l * random.uniform(0.85, 1.15)))
            },
            "Actual Work": {
                "Small": max(1, round(dur_aw_s * random.uniform(0.85, 1.15))),
                "Medium": max(1, round(dur_aw_m * random.uniform(0.85, 1.15))),
                "Large": max(1, round(dur_aw_l * random.uniform(0.85, 1.15)))
            }
        }
        # Randomize employee overhead by +/- 20%
        randomized_employee_cost = mean_employee_cost_base * random.uniform(0.80, 1.20)

        # --- The rest of the logic is now inside the loop ---
        if scenario_name == 'Personalized':
            pa_projects = []
            aw_projects = []
            # Use a different loop variable to avoid conflict
            for j in range(len(pa_s_all)):
                pa_projects.append({"Small": pa_s_all[j], "Medium": pa_m_all[j], "Large": pa_l_all[j]})
                aw_projects.append({"Small": aw_s_all[j], "Medium": aw_m_all[j], "Large": aw_l_all[j]})
        else:
            scenario = SCENARIOS[scenario_name]
            pa_projects, aw_projects = scenario['pa'], scenario['aw']
        
        custom_coeffs = {"Pre-analysis": {}, "Actual Work": {}}
        roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
        c_idx = 0
        for role in roles:
            custom_coeffs["Pre-analysis"][role] = {"Small": coeffs[c_idx], "Medium": coeffs[c_idx+1], "Large": coeffs[c_idx+2]}
            custom_coeffs["Actual Work"][role] = {"Small": coeffs[c_idx+3], "Medium": coeffs[c_idx+4], "Large": coeffs[c_idx+5]}
            c_idx += 6
        
        salaries = {role: salaries_tuple[i] for i, role in enumerate(roles)}
        
        custom_revenues = {
            "Pre-analysis": {"Small": revenues_tuple[0], "Medium": revenues_tuple[1], "Large": revenues_tuple[2]},
            "Actual Work": {"Small": revenues_tuple[3], "Medium": revenues_tuple[4], "Large": revenues_tuple[5]}
        }

        recurring_expenses = []
        recurring_revenues = []
        for i in range(len(entry_names)):
            name, start_str, end_str, value, entry_type = entry_names[i], entry_starts[i], entry_ends[i], entry_values[i], entry_types[i]
            if name and start_str and value:
                try:
                    start_date = pd.to_datetime(start_str, format='%Y-%m')
                    end_date = pd.to_datetime(end_str, format='%Y-%m') if end_str else pd.Timestamp.max
                    entry = {'start': start_date, 'end': end_date, 'value': value}
                    if entry_type == 'expense':
                        recurring_expenses.append(entry)
                    else:
                        recurring_revenues.append(entry)
                except ValueError:
                    pass 

        conversion_rate = conversion_rate_percent / 100.0

        simulation_df = run_simulation(
            all_years_pa_counts=pa_projects,
            all_years_aw_base_counts=aw_projects,
            conversion_rate=conversion_rate,
            simulation_duration_months=duration,
            project_durations=randomized_durations, # Use randomized durations
            staffing_coefficients=custom_coeffs 
        )

        simulation_df['Total Employee Cost'] = 0
        total_employees = 0
        for role in roles:
            total_hired_col = f'{role} Hired (Total)'
            simulation_df['Total Employee Cost'] += simulation_df[total_hired_col] * salaries[role]
            total_employees += simulation_df[total_hired_col]
        
        simulation_df['Total Employee Cost'] += total_employees * randomized_employee_cost # Use randomized overhead

        simulation_df['Recurring Expenses'] = 0
        for expense in recurring_expenses:
            mask = (simulation_df['Date'] >= expense['start']) & (simulation_df['Date'] <= expense['end'])
            simulation_df.loc[mask, 'Recurring Expenses'] += expense['value']

        simulation_df['Total Monthly Cost'] = simulation_df['Total Employee Cost'] + simulation_df['Recurring Expenses']
        
        revenue_map = {
            "PA Small": custom_revenues["Pre-analysis"]["Small"], "PA Medium": custom_revenues["Pre-analysis"]["Medium"], "PA Large": custom_revenues["Pre-analysis"]["Large"],
            "AW Small": custom_revenues["Actual Work"]["Small"], "AW Medium": custom_revenues["Actual Work"]["Medium"], "AW Large": custom_revenues["Actual Work"]["Large"]
        }
        shifted_projects = simulation_df[revenue_map.keys()].shift(1).fillna(0)
        simulation_df['Project Revenue'] = (shifted_projects * pd.Series(revenue_map)).sum(axis=1)

        simulation_df['Recurring Revenue'] = 0
        for revenue in recurring_revenues:
            mask = (simulation_df['Date'] >= revenue['start']) & (simulation_df['Date'] <= revenue['end'])
            simulation_df.loc[mask, 'Recurring Revenue'] += revenue['value']

        simulation_df['Monthly Revenue'] = simulation_df['Project Revenue'] + simulation_df['Recurring Revenue']
        simulation_df['Cash Flow'] = simulation_df['Monthly Revenue'] - simulation_df['Total Monthly Cost']
        simulation_df['Cumulative Cash Flow'] = simulation_df['Cash Flow'].cumsum()
        
        all_runs_cash_flow.append(simulation_df['Cumulative Cash Flow'])
        
        # Store the dataframe from each run, the last one will be used for representative graphs
        last_run_df = simulation_df

    # --- END OF MONTE CARLO LOOP ---

    # Process Monte Carlo results
    mc_df = pd.concat(all_runs_cash_flow, axis=1)
    mc_results = pd.DataFrame({
        'Date': last_run_df['Date'],
        'Median': mc_df.quantile(0.5, axis=1),
        'Lower Bound (5th percentile)': mc_df.quantile(0.05, axis=1),
        'Upper Bound (95th percentile)': mc_df.quantile(0.95, axis=1)
    })

    # Use the last run for the other graphs and totals (as a representative sample)
    total_project_cost = last_run_df['Total Monthly Cost'].sum()
    total_project_revenue = last_run_df['Monthly Revenue'].sum()
    total_cash_flow = last_run_df['Cash Flow'].sum()

    hired_total_columns = [col for col in last_run_df.columns if 'Hired (Total)' in col]
    mean_data = []
    for col in hired_total_columns:
        role_name = col.replace(" Hired (Total)", "")
        mean_value = last_run_df[col].mean()
        mean_data.append({
            "Role": role_name,
            "Mean Hired Employees": mean_value,
            "Mean Hired Employees Rounded Up": math.ceil(mean_value)
        })
    
    mean_results_df = pd.DataFrame(mean_data)
    mean_results_df["Mean Hired Employees Rounded Up"] = mean_results_df["Mean Hired Employees Rounded Up"].astype(int)
    
    try:
        mean_results_df.to_excel("results.xlsx", index=False, engine='openpyxl')
        confirmation_message = f"✅ Successfully exported mean results to 'results.xlsx' at {datetime.now().strftime('%H:%M:%S')}."
    except Exception as e:
        confirmation_message = f"❌ Error exporting to Excel: {e}. Please ensure 'openpyxl' is installed (`pip install openpyxl`)."

    # --- Create the graphs ---
    fig_employees = px.line(last_run_df, x='Date', y=hired_total_columns, title='Forecasted Hired Employees by Role (Representative Run with Mean)')
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(hired_total_columns):
        mean_value = last_run_df[col].mean()
        line_color = colors[i % len(colors)]
        fig_employees.add_shape(type="line", x0=last_run_df['Date'].min(), y0=mean_value, x1=last_run_df['Date'].max(), y1=mean_value, line=dict(color=line_color, width=2, dash="dash"))
        fig_employees.add_annotation(x=last_run_df['Date'].max(), y=mean_value, text=f"Mean: {mean_value:.1f}", showarrow=False, xshift=45, font=dict(color=line_color))
    fig_employees.for_each_trace(lambda t: t.update(name = t.name.replace(" Hired (Total)", "")))
    fig_employees.update_layout(legend_title_text='Employee Role', yaxis_title='Number of Employees')

    active_project_columns = ["PA Small", "PA Medium", "PA Large", "AW Small", "AW Medium", "AW Large"]
    fig_projects = px.bar(last_run_df, x='Date', y=active_project_columns, title='Active Projects by Type and Scale (Representative Run)')
    fig_projects.update_layout(legend_title_text='Project Type', yaxis_title='Number of Active Projects', barmode='stack')
    
    fig_mc_cash_flow = go.Figure([
        go.Scatter(
            name='Upper Bound (95%)', x=mc_results['Date'], y=mc_results['Upper Bound (95th percentile)'],
            mode='lines', line=dict(width=0.5, color='lightgrey')
        ),
        go.Scatter(
            name='Lower Bound (5%)', x=mc_results['Date'], y=mc_results['Lower Bound (5th percentile)'],
            mode='lines', line=dict(width=0.5, color='lightgrey'),
            fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty'
        ),
        go.Scatter(
            name='Median', x=mc_results['Date'], y=mc_results['Median'],
            mode='lines', line=dict(color='rgb(31, 119, 180)')
        ),
    ])
    fig_mc_cash_flow.update_layout(
        title=f'Cumulative Cash Flow Confidence Interval ({mc_runs} runs)',
        yaxis_title='Cumulative Cash Flow ($)',
        yaxis_tickprefix='$',
        showlegend=True
    )
    fig_mc_cash_flow.add_hline(y=0, line_dash="dash", line_color="grey")


    return html.Div([
        html.Div(confirmation_message, style={'textAlign': 'center', 'color': 'green' if 'Successfully' in confirmation_message else 'red', 'marginBottom': '15px', 'fontWeight': 'bold'}),
        
        html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}, children=[
            html.Div([
                html.H4("Total Revenue (Sample)", style={'textAlign': 'center', 'margin': '0'}),
                html.P(f"${total_project_revenue:,.2f}", style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'color': '#2ca02c', 'margin': '0'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'width': '30%'}),
            html.Div([
                html.H4("Total Cost (Sample)", style={'textAlign': 'center', 'margin': '0'}),
                html.P(f"${total_project_cost:,.2f}", style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'color': '#d62728', 'margin': '0'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'width': '30%'}),
            html.Div([
                html.H4("Net Cash Flow (Sample)", style={'textAlign': 'center', 'margin': '0'}),
                html.P(f"${total_cash_flow:,.2f}", style={'textAlign': 'center', 'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007BFF', 'margin': '0'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'width': '30%'}),
        ]),
        
        dcc.Graph(id='mc-cash-flow-graph', figure=fig_mc_cash_flow),
        html.Hr(),
        dcc.Graph(id='employees-graph', figure=fig_employees),
        html.Hr(),
        dcc.Graph(id='projects-graph', figure=fig_projects)
    ])

# --- Run the application ---
if __name__ == '__main__':
    app.run(debug=True)
