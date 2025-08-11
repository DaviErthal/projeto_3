# app.py
#
# A dynamic cash flow simulation tool based on teams.
# VERSION 5: Added a one-month lag for team revenue.

import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import copy

import dash
from dash import dcc, html, ALL, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# --- PHASE 1: Default Data & Initial Setup ---

DEFAULT_CREW_MEMBER = {"role": "New Role", "salary": 5000, "count": 1}

DEFAULT_PROJECT_TEAM = {
    "name": "Equipe: ",
    "duration": 24,
    "revenue": 357000,
    "crew": [
        {"role": "Gerente de Projeto", "salary": 13000, "count": 5},
        {"role": "Engenheiro Fiscal", "salary": 10312, "count": 5},
        {"role": "Técnico Fiscal", "salary": 3100, "count": 1},
        {"role": "Técnico de Automação/Eletricista", "salary": 3100, "count": 1},
        {"role": "Técnico de Segurança", "salary": 3100, "count": 1},
        {"role": "Engenheiro Eletricista/Automação", "salary": 10312, "count": 5},
        {"role": "Engenheiro Mecânico", "salary": 10312, "count": 5},
        {"role": "Engenheiro Planejador", "salary": 10312, "count":5},
    ],
}

INITIAL_STATE = {
    "fixed_team": {
        "name": "Equipe Fixa",
        "duration": 24,
        "revenue": 10000,
        "crew": [
            {"role": "Responsável Técnico", "salary": 35000, "count": 1},
            {"role": "Analista Administrativo/Técnico", "salary": 3000, "count": 1},
            {"role": "Engenheiro de Segurança", "salary": 10312, "count": 5},
            {"role": "Especialista em Administração Contratural", "salary": 8000, "count": 1},
        ],
    },
    "project_teams": [
        {
            "name": "Equipe 1 (Rio Manso)",
            "duration": 18,
            "revenue": 405000,
            "crew": [
                {"role": "Gerente de Projeto", "salary": 13000, "count": 5},
                {"role": "Engenheiro Fiscal", "salary": 10312, "count": 5},
                {"role": "Técnico Fiscal", "salary": 3100, "count": 1},
                {"role": "Técnico de Automação/Eletricista", "salary": 3100, "count": 1},
                {"role": "Técnico de Segurança", "salary": 3100, "count": 1},
                {"role": "Engenheiro Eletricista/Automação", "salary": 10312, "count": 5},
                {"role": "Engenheiro Mecânico", "salary": 10312, "count": 5},
                {"role": "Engenheiro Planejador", "salary": 10312, "count": 5},
            ],
        },
        {
            "name": "Equipe 2 (Bela Fama)",
            "duration": 24,
            "revenue": 405000,
            "crew": [
                {"role": "Gerente de Projeto", "salary": 13000, "count": 5},
                {"role": "Engenheiro Fiscal", "salary": 10312, "count": 5},
                {"role": "Técnico de Automação/Eletricista", "salary": 3100, "count": 1},
                {"role": "Técnico de Segurança", "salary": 3100, "count": 1},
                {"role": "Engenheiro Eletricista/Automação", "salary": 10312, "count": 5},
                {"role": "Engenheiro Mecânico", "salary": 10312, "count": 5},
                {"role": "Engenheiro Planejador", "salary": 10312, "count": 5},
            ],
        },
    ],
    "other_financials": [
        {"name": "Office Rent", "start": 1, "end": 24, "amount": -15000},
        {"name": "Cloud Services", "start": 1, "end": 24, "amount": -5000},
    ]
}


# --- PHASE 2: SIMULATION & FINANCIAL LOGIC ---

def run_simulation(fixed_team_data, project_teams_data, other_financials_data, start_date_str='2025-01-01'):
    all_teams = [fixed_team_data] + project_teams_data
    max_duration = 0
    # Ensure max_duration accounts for the revenue lag
    for team in all_teams:
        max_duration = max(max_duration, (team.get('duration', 0) or 0) + 1)
    for item in other_financials_data:
        max_duration = max(max_duration, (item.get('end', 0) or 0))
    if max_duration == 0: max_duration = 1 

    results_data = []
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    for month in range(1, max_duration + 1):
        current_date = start_date + pd.DateOffset(months=month - 1)
        monthly_revenue = 0
        monthly_cost = 0

        for team in all_teams:
            duration = team.get('duration', 0) or 0
            
            # --- MODIFIED LOGIC ---
            # 1. Costs are incurred while the team is active (Month 1 to Duration)
            if 1 <= month <= duration:
                for member in team.get('crew', []):
                    monthly_cost += (member.get('salary', 0) or 0) * (member.get('count', 0) or 0)
            
            # 2. Revenue is lagged by one month (Month 2 to Duration + 1)
            if 2 <= month <= duration + 1:
                monthly_revenue += (team.get('revenue', 0) or 0)
        
        for item in other_financials_data:
            if (item.get('start', 0) or 0) <= month <= (item.get('end', 0) or 0):
                amount = item.get('amount', 0) or 0
                if amount > 0: monthly_revenue += amount
                else: monthly_cost -= amount
        
        results_data.append({
            "Date": current_date,
            "Monthly Revenue": monthly_revenue,
            "Monthly Cost": monthly_cost,
            "Monthly Cash Flow": monthly_revenue - monthly_cost
        })

    results_df = pd.DataFrame(results_data)
    if not results_df.empty:
        results_df['Cumulative Cash Flow'] = results_df['Monthly Cash Flow'].cumsum()
    else:
        results_df = pd.DataFrame(columns=['Date', 'Monthly Revenue', 'Monthly Cost', 'Monthly Cash Flow', 'Cumulative Cash Flow'])

    return results_df


# --- PHASE 3: DASH APP LAYOUT & CALLBACKS ---

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Helper functions to create UI components ---

def create_role_input_row(team_id, role_id, data):
    return html.Div([
        dcc.Input(id={'type': 'role-input', 'team_id': team_id, 'role_id': role_id, 'field': 'role'}, type='text', value=data.get("role"), placeholder="Role Name", style={'width': '30%'}),
        dcc.Input(id={'type': 'role-input', 'team_id': team_id, 'role_id': role_id, 'field': 'salary'}, type='number', value=data.get("salary"), placeholder="Monthly Salary", style={'width': '25%'}),
        dcc.Input(id={'type': 'role-input', 'team_id': team_id, 'role_id': role_id, 'field': 'count'}, type='number', value=data.get("count"), placeholder="Count", min=0, step=1, style={'width': '15%'}),
        html.Button("X", id={'type': 'remove-role-btn', 'team_id': team_id, 'role_id': role_id}, className='remove-btn')
    ], id={'type': 'role-row', 'team_id': team_id, 'role_id': role_id}, className='role-row')

def create_team_card(team_id, data, is_fixed=False):
    team_type_str = 'fixed' if is_fixed else 'project'
    title = data.get("name") if is_fixed else dcc.Input(id={'type': 'team-input', 'team_id': team_id, 'field': 'name'}, value=data.get("name"))
    
    return html.Div([
        html.Div([
            html.H4(title, style={'flexGrow': 2}),
            html.Button("X", id={'type': 'remove-team-btn', 'team_id': team_id}, className='remove-btn') if not is_fixed else ""
        ], className='card-header'),
        html.Div([
            html.Label("Duration (Months):"), 
            dcc.Input(id={'type': 'team-input', 'team_id': team_id, 'field': 'duration'}, type='number', value=data.get("duration"), min=1, step=1, style={'width': '80px'}),
            html.Label("Monthly Revenue:"), 
            dcc.Input(id={'type': 'team-input', 'team_id': team_id, 'field': 'revenue'}, type='number', value=data.get("revenue"), style={'width': '100px'}),
        ], className='card-controls'),
        html.H5("Crew Composition"),
        html.Div(id={'type': f'{team_type_str}-roles-container', 'team_id': team_id}),
        html.Button("Add Role", id={'type': 'add-role-btn', 'team_id': team_id}, className='add-btn')
    ], id={'type': f'{team_type_str}-team-card', 'team_id': team_id}, className='team-card')

# --- App Layout ---
app.layout = html.Div([
    html.H1("Team-Based Cash Flow Modeler", style={'textAlign': 'center'}),
    
    dcc.Store(id='fixed-team-store', data=INITIAL_STATE["fixed_team"]),
    dcc.Store(id='project-teams-store', data=INITIAL_STATE["project_teams"]),
    
    html.Div([
        html.Div([
            html.H3("Fixed Team"),
            html.Div(id='fixed-team-container')
        ], className='control-section'),

        html.Div([
            html.H3("Project Teams"),
            html.Div(id='project-teams-container'),
            html.Button("Add Project Team", id='add-project-team-btn', n_clicks=0, className='add-btn')
        ], className='control-section'),
    ], className='control-panel'),
    
    html.Div(
        html.Button('Run Simulation', id='run-button', n_clicks=0, className='run-btn'),
        style={'textAlign': 'center', 'margin': '20px'}
    ),
    
    dcc.Loading(id="loading-icon", children=[html.Div(id='output-graphs')], type="circle"),

], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# --- Callbacks for Dynamic UI ---

# RENDER UI from stores
@app.callback(Output('fixed-team-container', 'children'), Input('fixed-team-store', 'data'))
def render_fixed_team(data):
    if not data: return []
    return create_team_card('fixed-0', data, is_fixed=True)

@app.callback(Output({'type': 'fixed-roles-container', 'team_id': 'fixed-0'}, 'children'), Input('fixed-team-store', 'data'))
def render_fixed_team_roles(data):
    if not data: return []
    return [create_role_input_row('fixed-0', i, role_data) for i, role_data in enumerate(data.get("crew", []))]

@app.callback(Output('project-teams-container', 'children'), Input('project-teams-store', 'data'))
def render_project_teams(teams_data):
    return [create_team_card(f'project-{i}', data) for i, data in enumerate(teams_data)]

@app.callback(Output({'type': 'project-roles-container', 'team_id': MATCH}, 'children'), Input('project-teams-store', 'data'), State({'type': 'project-roles-container', 'team_id': MATCH}, 'id'))
def render_project_team_roles(teams_data, team_id_dict):
    team_index = int(team_id_dict['team_id'].split('-')[-1])
    if team_index < len(teams_data):
        return [create_role_input_row(team_id_dict['team_id'], i, role_data) for i, role_data in enumerate(teams_data[team_index].get("crew", []))]
    return []

# ADD/REMOVE Project Teams
@app.callback(
    Output('project-teams-store', 'data', allow_duplicate=True),
    Input('add-project-team-btn', 'n_clicks'),
    State('project-teams-store', 'data'),
    prevent_initial_call=True
)
def add_project_team(n_clicks, teams_data):
    new_team = copy.deepcopy(DEFAULT_PROJECT_TEAM)
    new_team["name"] = f"Project Team {len(teams_data) + 1}"
    teams_data.append(new_team)
    return teams_data

@app.callback(
    Output('project-teams-store', 'data', allow_duplicate=True),
    Input({'type': 'remove-team-btn', 'team_id': ALL}, 'n_clicks'),
    State('project-teams-store', 'data'),
    prevent_initial_call=True
)
def remove_project_team(remove_clicks, teams_data):
    ctx = dash.callback_context
    if not any(c and c > 0 for c in remove_clicks):
        raise PreventUpdate
    
    team_id_to_remove = ctx.triggered_id['team_id']
    team_index_to_remove = int(team_id_to_remove.split('-')[-1])
    
    if 0 <= team_index_to_remove < len(teams_data):
        teams_data.pop(team_index_to_remove)
    
    return teams_data


# ADD/REMOVE roles
@app.callback(
    Output('fixed-team-store', 'data', allow_duplicate=True),
    Output('project-teams-store', 'data', allow_duplicate=True),
    Input({'type': 'add-role-btn', 'team_id': ALL}, 'n_clicks'),
    Input({'type': 'remove-role-btn', 'team_id': ALL, 'role_id': ALL}, 'n_clicks'),
    State('fixed-team-store', 'data'),
    State('project-teams-store', 'data'),
    prevent_initial_call=True
)
def update_roles(add_clicks, remove_clicks, fixed_data, projects_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    
    if 'add-role-btn' in str(triggered_id):
        team_id = triggered_id['team_id']
        if team_id.startswith('fixed'):
            fixed_data.setdefault('crew', []).append(copy.deepcopy(DEFAULT_CREW_MEMBER))
        else:
            team_index = int(team_id.split('-')[-1])
            if team_index < len(projects_data):
                projects_data[team_index].setdefault('crew', []).append(copy.deepcopy(DEFAULT_CREW_MEMBER))
    
    elif 'remove-role-btn' in str(triggered_id):
        team_id = triggered_id['team_id']
        role_id = triggered_id['role_id']
        if team_id.startswith('fixed'):
            if 'crew' in fixed_data and 0 <= role_id < len(fixed_data['crew']):
                fixed_data['crew'].pop(role_id)
        else:
            team_index = int(team_id.split('-')[-1])
            if team_index < len(projects_data) and 'crew' in projects_data[team_index] and 0 <= role_id < len(projects_data[team_index]['crew']):
                projects_data[team_index]['crew'].pop(role_id)

    return fixed_data, projects_data


# --- Main Callback to Run Simulation ---
@app.callback(
    Output('output-graphs', 'children'),
    Input('run-button', 'n_clicks'),
    State('fixed-team-store', 'data'),
    State('project-teams-store', 'data'),
    State({'type': 'team-input', 'team_id': ALL, 'field': ALL}, 'value'),
    State({'type': 'team-input', 'team_id': ALL, 'field': ALL}, 'id'),
    State({'type': 'role-input', 'team_id': ALL, 'role_id': ALL, 'field': ALL}, 'value'),
    State({'type': 'role-input', 'team_id': ALL, 'role_id': ALL, 'field': ALL}, 'id'),
)
def update_dashboard(n_clicks, fixed_team_data, project_teams_data, team_values, team_ids, role_values, role_ids):
    if n_clicks == 0:
        return html.Div("Click 'Run Simulation' to see the results.", style={'textAlign': 'center', 'padding': '50px'})

    fixed_team = copy.deepcopy(fixed_team_data)
    project_teams = copy.deepcopy(project_teams_data)
    other_financials = copy.deepcopy(INITIAL_STATE['other_financials'])

    for value, component_id in zip(team_values, team_ids):
        if value is None: continue
        team_id_str = component_id['team_id']
        field = component_id['field']
        
        if team_id_str.startswith('fixed'):
            fixed_team[field] = value
        elif team_id_str.startswith('project'):
            team_index = int(team_id_str.split('-')[-1])
            if team_index < len(project_teams):
                project_teams[team_index][field] = value

    for value, component_id in zip(role_values, role_ids):
        if value is None: continue
        team_id_str = component_id['team_id']
        role_index = component_id['role_id']
        field = component_id['field']

        if team_id_str.startswith('fixed'):
            if 'crew' in fixed_team and role_index < len(fixed_team['crew']):
                fixed_team['crew'][role_index][field] = value
        elif team_id_str.startswith('project'):
            team_index = int(team_id_str.split('-')[-1])
            if team_index < len(project_teams) and 'crew' in project_teams[team_index] and role_index < len(project_teams[team_index]['crew']):
                project_teams[team_index]['crew'][role_index][field] = value

    sim_df = run_simulation(fixed_team, project_teams, other_financials)

    if sim_df.empty:
        return html.Div("No data to display.", style={'textAlign': 'center', 'padding': '50px'})

    total_revenue = sim_df['Monthly Revenue'].sum()
    total_cost = sim_df['Monthly Cost'].sum()
    net_cash_flow = sim_df['Monthly Cash Flow'].sum()

    fig_monthly_cash_flow = px.bar(
        sim_df, x='Date', y='Monthly Cash Flow', title="Monthly Cash Flow",
        color='Monthly Cash Flow', color_continuous_scale=['#d62728', '#2ca02c']
    )
    fig_monthly_cash_flow.update_layout(coloraxis_showscale=False)

    fig_cumulative_cash_flow = px.area(
        sim_df, x='Date', y='Cumulative Cash Flow', title="Cumulative Cash Flow"
    )
    fig_cumulative_cash_flow.add_hline(y=0, line_dash="dash", line_color="grey")

    return html.Div([
        html.Div([
            html.Div([html.H4("Total Revenue"), html.P(f"${total_revenue:,.2f}")], className='metric-card green'),
            html.Div([html.H4("Total Cost"), html.P(f"${total_cost:,.2f}")], className='metric-card red'),
            html.Div([html.H4("Net Cash Flow"), html.P(f"${net_cash_flow:,.2f}")], className='metric-card blue'),
        ], className='metrics-container'),
        dcc.Graph(figure=fig_monthly_cash_flow),
        dcc.Graph(figure=fig_cumulative_cash_flow)
    ])


if __name__ == '__main__':
    app.run(debug=True)
