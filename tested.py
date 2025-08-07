# app.py
#
# This script creates an interactive web-based dashboard for the workforce simulation
# using the Dash framework by Plotly.
#
# NEW: Added cost analysis features, including salary inputs, a monthly cost graph,
# and a total cost summary.

import math
import random
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# You will need to install dash, pandas, plotly, and openpyxl
# pip install dash pandas plotly openpyxl
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# --- PHASE 1: SIMULATION LOGIC (Copied from our notebook) ---

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

# --- Default Monthly Salaries ---
DEFAULT_SALARIES = {
    "Analyst": 10312,
    "Technical": 3100,
    "Project Manager": 13000,
    "Tax Engineer": 10312,
    "Tax Technical": 3100,
    "Planning Engineer": 10312
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

# --- Project Class Definition ---
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

# --- Function to run the simulation ---
def run_simulation(all_years_pa_counts, all_years_aw_base_counts, conversion_rate, simulation_duration_months, project_durations, staffing_coefficients, start_date_str='2026-01-01'):
    project_pipeline, project_id_counter, results_data = [], 0, []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    scales = ["Small", "Medium", "Large"]
    # Initialize the project pipeline with projects based on the yearly counts
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
                html.Label("PA Small:"), dcc.Input(id=f'pa-small-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("PA Medium:"), dcc.Input(id=f'pa-medium-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("PA Large:"), dcc.Input(id=f'pa-large-y{year}', type='number', value=0, style={'width': '80px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '10px'}),
            html.Div([
                html.Label("AW Small:"), dcc.Input(id=f'aw-small-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("AW Medium:"), dcc.Input(id=f'aw-medium-y{year}', type='number', value=0, style={'width': '80px'}),
                html.Label("AW Large:"), dcc.Input(id=f'aw-large-y{year}', type='number', value=0, style={'width': '80px'}),
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

# --- Helper function to create salary inputs ---
def create_cost_inputs():
    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    inputs = []
    for role in roles:
        sanitized_role = role.replace(" ", "").lower()
        inputs.append(html.Label(f"{role} Salary:"))
        inputs.append(dcc.Input(id=f'salary-{sanitized_role}', type='number', value=DEFAULT_SALARIES.get(role, 0), style={'width': '100px', 'marginRight': '20px'}))
    return html.Div(inputs, style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})

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
            create_year_inputs(1), create_year_inputs(2),
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
                # --- ADDED: Collapsible section for cost inputs ---
                html.Details([
                    html.Summary('Customize Costs (Monthly Salary)'),
                    html.Div(create_cost_inputs(), style={'marginTop': '10px'})
                ])
            ])
        ], style={'marginTop': '20px'})
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
    Output('output-graphs', 'children'),
    Input('run-button', 'n_clicks'),
    [
        State('scenario-dropdown', 'value'), State('conversion-rate-slider', 'value'), State('duration-input', 'value'),
        # Personalized scenario inputs
        State('pa-small-y1', 'value'), State('pa-medium-y1', 'value'), State('pa-large-y1', 'value'),
        State('aw-small-y1', 'value'), State('aw-medium-y1', 'value'), State('aw-large-y1', 'value'),
        State('pa-small-y2', 'value'), State('pa-medium-y2', 'value'), State('pa-large-y2', 'value'),
        State('aw-small-y2', 'value'), State('aw-medium-y2', 'value'), State('aw-large-y2', 'value'),
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
        # --- States for salary inputs ---
        State('salary-analyst', 'value'), State('salary-technical', 'value'), State('salary-projectmanager', 'value'),
        State('salary-taxengineer', 'value'), State('salary-taxtechnical', 'value'), State('salary-planningengineer', 'value'),
    ]
)
def update_dashboard(n_clicks, scenario_name, conversion_rate_percent, duration,
                     pa_s1, pa_m1, pa_l1, aw_s1, aw_m1, aw_l1,
                     pa_s2, pa_m2, pa_l2, aw_s2, aw_m2, aw_l2,
                     dur_pa_s, dur_pa_m, dur_pa_l, dur_aw_s, dur_aw_m, dur_aw_l,
                     *coeffs_and_salaries): 
    if n_clicks == 0:
        return html.Div("Please set your parameters and click 'Run Simulation'.", style={'textAlign': 'center', 'padding': '50px', 'fontSize': '18px'})

    # Separate coeffs and salaries from the combined tuple
    coeffs = coeffs_and_salaries[:36]
    salaries_tuple = coeffs_and_salaries[36:]

    if scenario_name == 'Personalized':
        pa_projects = [{"Small": pa_s1, "Medium": pa_m1, "Large": pa_l1}, {"Small": pa_s2, "Medium": pa_m2, "Large": pa_l2}]
        aw_projects = [{"Small": aw_s1, "Medium": aw_m1, "Large": aw_l1}, {"Small": aw_s2, "Medium": aw_m2, "Large": aw_l2}]
    else:
        scenario = SCENARIOS[scenario_name]
        pa_projects, aw_projects = scenario['pa'], scenario['aw']
    
    custom_durations = {
        "Pre-analysis": {"Small": dur_pa_s, "Medium": dur_pa_m, "Large": dur_pa_l},
        "Actual Work": {"Small": dur_aw_s, "Medium": dur_aw_m, "Large": dur_aw_l}
    }

    custom_coeffs = {"Pre-analysis": {}, "Actual Work": {}}
    roles = ["Analyst", "Technical", "Project Manager", "Tax Engineer", "Tax Technical", "Planning Engineer"]
    c_idx = 0
    for role in roles:
        custom_coeffs["Pre-analysis"][role] = {"Small": coeffs[c_idx], "Medium": coeffs[c_idx+1], "Large": coeffs[c_idx+2]}
        custom_coeffs["Actual Work"][role] = {"Small": coeffs[c_idx+3], "Medium": coeffs[c_idx+4], "Large": coeffs[c_idx+5]}
        c_idx += 6
    
    # --- Reconstruct salaries dictionary ---
    salaries = {role: salaries_tuple[i] for i, role in enumerate(roles)}

    conversion_rate = conversion_rate_percent / 100.0

    simulation_df = run_simulation(
        all_years_pa_counts=pa_projects,
        all_years_aw_base_counts=aw_projects,
        conversion_rate=conversion_rate,
        simulation_duration_months=duration,
        project_durations=custom_durations,
        staffing_coefficients=custom_coeffs 
    )

    # --- Cost Calculation ---
    simulation_df['Total Monthly Cost'] = 0
    for role in roles:
        simulation_df['Total Monthly Cost'] += simulation_df[f'{role} Hired (Total)'] * salaries[role]
    
    total_project_cost = simulation_df['Total Monthly Cost'].sum()


    # --- EXPORT TO EXCEL ---
    hired_total_columns = [col for col in simulation_df.columns if 'Hired (Total)' in col]
    mean_data = []
    for col in hired_total_columns:
        role_name = col.replace(" Hired (Total)", "")
        mean_value = simulation_df[col].mean()
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
    fig_employees = px.line(simulation_df, x='Date', y=hired_total_columns, title='Forecasted Hired Employees by Role (with Mean)')
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(hired_total_columns):
        mean_value = simulation_df[col].mean()
        line_color = colors[i % len(colors)]
        fig_employees.add_shape(type="line", x0=simulation_df['Date'].min(), y0=mean_value, x1=simulation_df['Date'].max(), y1=mean_value, line=dict(color=line_color, width=2, dash="dash"))
        fig_employees.add_annotation(x=simulation_df['Date'].max(), y=mean_value, text=f"Mean: {mean_value:.1f}", showarrow=False, xshift=45, font=dict(color=line_color))
    fig_employees.for_each_trace(lambda t: t.update(name = t.name.replace(" Hired (Total)", "")))
    fig_employees.update_layout(legend_title_text='Employee Role', yaxis_title='Number of Employees')

    active_project_columns = ["PA Small", "PA Medium", "PA Large", "AW Small", "AW Medium", "AW Large"]
    fig_projects = px.bar(simulation_df, x='Date', y=active_project_columns, title='Active Projects by Type and Scale Over Time')
    fig_projects.update_layout(legend_title_text='Project Type', yaxis_title='Number of Active Projects', barmode='stack')
    
    # --- Cost Graph ---
    fig_cost = px.area(simulation_df, x='Date', y='Total Monthly Cost', title='Total Monthly Payroll Cost Over Time')
    fig_cost.update_layout(yaxis_title='Total Cost ($)', yaxis_tickprefix='$')


    return html.Div([
        html.Div(confirmation_message, style={'textAlign': 'center', 'color': 'green' if 'Successfully' in confirmation_message else 'red', 'marginBottom': '15px', 'fontWeight': 'bold'}),
        
        # --- Total Cost Summary Card ---
        html.Div([
            html.H3("Total Simulation Cost", style={'textAlign': 'center', 'margin': '0'}),
            html.P(f"${total_project_cost:,.2f}", style={'textAlign': 'center', 'fontSize': '28px', 'fontWeight': 'bold', 'color': '#007BFF', 'margin': '0'})
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        dcc.Graph(id='cost-graph', figure=fig_cost),
        html.Hr(),
        dcc.Graph(id='employees-graph', figure=fig_employees),
        html.Hr(),
        dcc.Graph(id='projects-graph', figure=fig_projects)
    ])

# --- Run the application ---
if __name__ == '__main__':
    app.run(debug=True)
