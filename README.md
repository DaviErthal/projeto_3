# Workforce Demand & Optimization Simulator

**Version:** 1.0
**Last Updated:** 2025-07-23
**Author:** Gemini AI & User Collaboration

---

## 1. Project Overview

This project provides a dynamic simulation tool, written in Python, to forecast staffing needs for a construction or engineering company. It addresses the core business problem of minimizing labor costs by accurately predicting the number of employees required for various roles based on a pipeline of incoming projects.

The model moves beyond simple averages by incorporating key business logic, including:
- A multi-stage project lifecycle ("Pre-analysis" converting to "Actual Work").
- Project classification by scale (Small, Medium, Large).
- Employee specialization, where staff are dedicated to projects of a specific scale.
- A configurable conversion rate to model business success in winning contracts.

The final output is a detailed, month-by-month time-series DataFrame, perfect for analysis, visualization, and strategic planning.

---

## 2. Core Features

- **Dynamic Project Generation:** Automatically generates a project pipeline based on yearly contract estimates for multiple years.
- **Lifecycle Simulation:** Models the transition of projects from a preliminary analysis phase to actual work based on a defined conversion rate.
- **Rule-Based Staffing:** Calculates workforce demand using specific, configurable "staffing coefficients" (e.g., "1 Tax Engineer per 3 Large projects").
- **Employee Specialization:** Accurately models the real-world constraint that employees working on large-scale projects cannot simultaneously work on small-scale ones, preventing over-optimistic pooling.
- **Structured Output:** Exports the complete simulation run into a clean `pandas` DataFrame, ready for use in notebooks for visualization (`plotly`) or further analysis.
- **Time-Series Analysis:** The output is designed for time-series plotting, allowing for easy visualization of demand peaks and troughs over time.

---

## 3. How to Use

This simulator is designed to be run within a Jupyter Notebook or a similar environment.

**Step 1: Setup**
- Ensure you have Python installed with the `pandas` library. If not, run:
  ```bash
  pip install pandas

- The main simulation logic is contained in the `main.ipynb`.

**Step 2: Configure the Simulation**

At the first notebook cell you can configure the simulation parameters:

- `ALL_YEARS_PA_PROJECTS`: A list of dictionaries defining the number of new "Pre-analysis" projects expected each year.

- `ALL_YEARS_AW_BASE_PROJECTS`: A list of dictionaries for new "Actual Work" projects that don't come from conversions.

- `CONVERSION_RATE`: A float between 0.0 and 1.0 representing the probability that a "Pre-analysis" project becomes "Actual Work".

- `simulation_duration_months`: The total number of months to forecast.

- `start_date_str`: The starting date for the simulation report (e.g., '2026-01-01').

**Step 3: Run the Simulation**

- Execute the cell containing the run_simulation function. It will return a pandas DataFrame.

**Step 4: Analyze and Visualize**

- Use the returned simulation_df DataFrame in subsequent cells for analysis or plotting with libraries like `plotly` or `matplotlib`.


