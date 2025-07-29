# app.py
#
# This script creates an interactive web-based dashboard for the workforce simulation
# using the Dash framework by Plotly.
#
# NEW: Added an advanced settings section to customize project durations.

import math
import random
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# You will need to install dash and its components
# pip install dash pandas plotly
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# --- PHASE 1: SIMULATION LOGIC (Copied from our notebook) ---
# We keep all the core simulation logic the same.

# Rule 1: Project Durations in months
PROJECT_DURATIONS = {
    "Pre-analysis": { "Small": 3, "Medium": 6, "Large": 8 },
    "Actual Work": { "Small": 11, "Medium": 20, "Large": 25 }
}

# Rule 2: Staffing Coefficients
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

# --- Data Scenarios ---
# We define our different project scenarios in a dictionary for easy access.
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
    # ADDED: A key for the personalized option
    "Personalized": {}
}


class Project:
    def __init__(self, id, p_type, p_scale, start_month, project_durations):
        self.id, self.p_type, self.p_scale, self.start_month = id, p_type, p_scale, start_month
        self.duration = project_durations[p_type][p_scale]
        self.end_month = start_month + self.duration - 1
    def is_active_in_month(self, month):
        return self.start_month <= month <= self.end_month
    def get_staffing_needs(self):
        return STAFFING_COEFFICIENTS.get(self.p_type, {})

def run_simulation(all_years_pa_counts, all_years_aw_base_counts, conversion_rate, simulation_duration_months, project_durations, start_date_str='2026-01-01'):
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
                    project_pipeline.append(Project(f"PA_{p_scale}_{project_id_counter}", "Pre-analysis", p_scale, month, project_durations))
            for p_scale, yearly_count in yearly_aw_base_counts.items():
                num_to_add = yearly_count // 12 + (1 if month_within_year < yearly_count % 12 else 0)
                for _ in range(num_to_add):
                    project_id_counter += 1
                    project_pipeline.append(Project(f"AW-base_{p_scale}_{project_id_counter}", "Actual Work", p_scale, month, project_durations))

        converted_this_month = 0
        finished_pa_projects = [p for p in project_pipeline if p.p_type == "Pre-analysis" and p.end_month == month - 1]
        for proj in finished_pa_projects:
            if random.random() < conversion_rate:
                converted_this_month += 1
                project_id_counter += 1
                project_pipeline.append(Project(f"AW-Conv_{proj.p_scale}_{project_id_counter}", "Actual Work", proj.p_scale, month, project_durations))

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

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server # Expose server for deployment

# --- ADDED: Helper function to create input fields for the personalized scenario ---
def create_year_inputs(year):
    return html.Div([
        html.H4(f"Year {year}", style={'marginTop': '20px'}),
        html.Div([
            html.Div([
                html.Label("PA Small:"),
                dcc.Input(id=f'pa-small-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("PA Medium:"),
                dcc.Input(id=f'pa-medium-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("PA Large:"),
                dcc.Input(id=f'pa-large-y{year}', type='number', value=0, style={'width': '80px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
            html.Div([
                html.Label("AW Small:"),
                dcc.Input(id=f'aw-small-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("AW Medium:"),
                dcc.Input(id=f'aw-medium-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("AW Large:"),
                dcc.Input(id=f'aw-large-y{year}', type='number', value=0, style={'width': '80px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        ])
    ])

# Define the layout of the application
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Workforce Planning & Optimization Dashboard", style={'textAlign': 'center', 'color': '#333'}),
    html.P("Use the controls below to run different scenarios and forecast staffing needs.", style={'textAlign': 'center', 'color': '#666'}),

    # Control Panel
    html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-around'}, children=[
            # Scenario Dropdown
            html.Div([
                html.Label("Select Project Scenario:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=[{'label': key, 'value': key} for key in SCENARIOS.keys()],
                    value='Lote 1', # Default value
                    clearable=False
                )
            ], style={'width': '30%'}),

            # Conversion Rate Slider
            html.Div([
                html.Label("Conversion Rate (%):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='conversion-rate-slider',
                    min=0, max=100, step=5, value=50,
                    marks={i: f'{i}%' for i in range(0, 101, 10)}
                )
            ], style={'width': '30%'}),

            # Duration Input
            html.Div([
                html.Label("Simulation Duration (Months):", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='duration-input',
                    type='number',
                    value=24,
                    min=12,
                    step=1
                )
            ], style={'width': '20%'})
        ]),
        
        # --- ADDED: Container for personalized inputs, initially hidden ---
        html.Div(id='personalized-inputs-container', style={'display': 'none', 'marginTop': '20px', 'borderTop': '1px solid #ccc', 'paddingTop': '20px'}, children=[
            html.H3("Enter Personalized Scenario Data", style={'textAlign': 'center'}),
            create_year_inputs(1),
            create_year_inputs(2),
        ]),
        
        # --- ADDED: Collapsible section for advanced settings ---
        html.Details([
            html.Summary('Advanced Settings (Customize Durations)'),
            html.Div(style={'marginTop': '10px'}, children=[
                html.Div([
                    html.Label("PA Small Duration:"),
                    dcc.Input(id='dur-pa-small', type='number', value=PROJECT_DURATIONS["Pre-analysis"]["Small"], style={'width': '80px'}),
                    html.Label("PA Medium Duration:"),
                    dcc.Input(id='dur-pa-medium', type='number', value=PROJECT_DURATIONS["Pre-analysis"]["Medium"], style={'width': '80px'}),
                    html.Label("PA Large Duration:"),
                    dcc.Input(id='dur-pa-large', type='number', value=PROJECT_DURATIONS["Pre-analysis"]["Large"], style={'width': '80px'}),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
                html.Div([
                    html.Label("AW Small Duration:"),
                    dcc.Input(id='dur-aw-small', type='number', value=PROJECT_DURATIONS["Actual Work"]["Small"], style={'width': '80px'}),
                    html.Label("AW Medium Duration:"),
                    dcc.Input(id='dur-aw-medium', type='number', value=PROJECT_DURATIONS["Actual Work"]["Medium"], style={'width': '80px'}),
                    html.Label("AW Large Duration:"),
                    dcc.Input(id='dur-aw-large', type='number', value=PROJECT_DURATIONS["Actual Work"]["Large"], style={'width': '80px'}),
                ], style={'display': 'flex', 'justifyContent': 'space-around'}),
            ])
        ], style={'marginTop': '20px'})
    ]),

    # Run Button
    html.Div(style={'textAlign': 'center', 'marginBottom': '20px'}, children=[
        html.Button('Run Simulation', id='run-button', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px'})
    ]),
    
    dcc.Loading(id="loading-icon", children=[html.Div(id='output-graphs')], type="circle"),
])

# --- PHASE 3: CALLBACKS TO CONNECT UI TO SIMULATION ---

# --- ADDED: Callback to show/hide the personalized input fields ---
@app.callback(
    Output('personalized-inputs-container', 'style'),
    Input('scenario-dropdown', 'value')
)
def toggle_personalized_inputs(scenario_name):
    if scenario_name == 'Personalized':
        return {'display': 'block', 'marginTop': '20px', 'borderTop': '1px solid #ccc', 'paddingTop': '20px'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('output-graphs', 'children'),
    Input('run-button', 'n_clicks'),
    [
        State('scenario-dropdown', 'value'),
        State('conversion-rate-slider', 'value'),
        State('duration-input', 'value'),
        # --- ADDED: States for all personalized input fields ---
        State('pa-small-y1', 'value'), State('pa-medium-y1', 'value'), State('pa-large-y1', 'value'),
        State('aw-small-y1', 'value'), State('aw-medium-y1', 'value'), State('aw-large-y1', 'value'),
        State('pa-small-y2', 'value'), State('pa-medium-y2', 'value'), State('pa-large-y2', 'value'),
        State('aw-small-y2', 'value'), State('aw-medium-y2', 'value'), State('aw-large-y2', 'value'),
        # --- ADDED: States for custom duration inputs ---
        State('dur-pa-small', 'value'), State('dur-pa-medium', 'value'), State('dur-pa-large', 'value'),
        State('dur-aw-small', 'value'), State('dur-aw-medium', 'value'), State('dur-aw-large', 'value'),
    ]
)
def update_dashboard(n_clicks, scenario_name, conversion_rate_percent, duration,
                     pa_s1, pa_m1, pa_l1, aw_s1, aw_m1, aw_l1,
                     pa_s2, pa_m2, pa_l2, aw_s2, aw_m2, aw_l2,
                     dur_pa_s, dur_pa_m, dur_pa_l, dur_aw_s, dur_aw_m, dur_aw_l):
    if n_clicks == 0:
        return html.Div("Please set your parameters and click 'Run Simulation'.", style={'textAlign': 'center', 'padding': '50px', 'fontSize': '18px'})

    # --- MODIFIED: Logic to handle personalized scenario ---
    if scenario_name == 'Personalized':
        # Build the scenario data from the user inputs
        pa_projects = [
            {"Small": pa_s1, "Medium": pa_m1, "Large": pa_l1},
            {"Small": pa_s2, "Medium": pa_m2, "Large": pa_l2}
        ]
        aw_projects = [
            {"Small": aw_s1, "Medium": aw_m1, "Large": aw_l1},
            {"Small": aw_s2, "Medium": aw_m2, "Large": aw_l2}
        ]
    else:
        # Get the selected predefined scenario data
        scenario = SCENARIOS[scenario_name]
        pa_projects = scenario['pa']
        aw_projects = scenario['aw']
    
    # --- ADDED: Create the durations dictionary from user inputs ---
    custom_durations = {
        "Pre-analysis": {"Small": dur_pa_s, "Medium": dur_pa_m, "Large": dur_pa_l},
        "Actual Work": {"Small": dur_aw_s, "Medium": dur_aw_m, "Large": dur_aw_l}
    }
    
    conversion_rate = conversion_rate_percent / 100.0

    simulation_df = run_simulation(
        all_years_pa_counts=pa_projects,
        all_years_aw_base_counts=aw_projects,
        conversion_rate=conversion_rate,
        simulation_duration_months=duration,
        project_durations=custom_durations # Pass the custom durations to the simulation
    )

    # --- Create the graphs (logic remains the same) ---
    hired_total_columns = [col for col in simulation_df.columns if 'Hired (Total)' in col]
    fig_employees = px.line(
        simulation_df, x='Date', y=hired_total_columns,
        title='Forecasted Hired Employees by Role (with Mean)'
    )
    
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(hired_total_columns):
        mean_value = simulation_df[col].mean()
        line_color = colors[i % len(colors)]
        fig_employees.add_shape(
            type="line",
            x0=simulation_df['Date'].min(), y0=mean_value,
            x1=simulation_df['Date'].max(), y1=mean_value,
            line=dict(color=line_color, width=2, dash="dash")
        )
        fig_employees.add_annotation(
            x=simulation_df['Date'].max(), y=mean_value,
            text=f"Mean: {mean_value:.1f}", showarrow=False,
            xshift=45, font=dict(color=line_color)
        )

    fig_employees.for_each_trace(lambda t: t.update(name = t.name.replace(" Hired (Total)", "")))
    fig_employees.update_layout(legend_title_text='Employee Role', yaxis_title='Number of Employees')

    active_project_columns = ["PA Small", "PA Medium", "PA Large", "AW Small", "AW Medium", "AW Large"]
    fig_projects = px.bar(
        simulation_df, x='Date', y=active_project_columns,
        title='Active Projects by Type and Scale Over Time'
    )
    fig_projects.update_layout(legend_title_text='Project Type', yaxis_title='Number of Active Projects', barmode='stack')

    return html.Div([
        dcc.Graph(id='employees-graph', figure=fig_employees),
        html.Hr(),
        dcc.Graph(id='projects-graph', figure=fig_projects)
    ])

# --- Run the application ---
if __name__ == '__main__':
    app.run(debug=True)
