#!/usr/bin/env python3

import csv
import os

def update_table3():
    """Update table3_latex.tex with data from updated_table3.csv"""
    csv_file = "updated_tables/updated_table3.csv"
    tex_file = "updated_tables/table3_latex.tex"
    
    # Read CSV data
    rows = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Create LaTeX table content
    tex_content = """\\begin{table}[h]
\\centering
\\caption{Comparison of direct classification vs. two-stage approach using Macro F1 and Micro F1 metrics}
\\label{tab:categorical_mapping}
\\begin{tabular}{|l|c|c|c|c|c|c|c|}
\\hline
\\textbf{Approach} & \\textbf{Modality} & \\textbf{Train Acc.} & \\textbf{Test Acc.} & \\textbf{Macro F1} & \\textbf{Micro F1} & \\textbf{Precision} & \\textbf{Recall} \\\\
\\hline
"""
    
    for row in rows:
        tex_content += f"{row['Approach']} & {row['Modality']} & {row['Train Accuracy']} & {row['Test Accuracy']} & {row['Macro F1']} & {row['Micro F1']} & {row['Precision']} & {row['Recall']} \\\\\n\\hline\n"
    
    tex_content += "\\end{tabular}\n\\end{table}\n"
    
    # Write to file
    with open(tex_file, "w") as f:
        f.write(tex_content)
    
    print(f"Updated {tex_file}")

def update_table4():
    """Update table4_latex.tex with data from updated_table4.csv"""
    csv_file = "updated_tables/updated_table4.csv"
    tex_file = "updated_tables/table4_latex.tex"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Skipping table4 update.")
        return
    
    # Read CSV data
    rows = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Create LaTeX table content
    tex_content = """\\begin{table}[h]
\\centering
\\caption{Comparison of our approaches with previous state-of-the-art results on the IEMOCAP dataset}
\\label{tab:sota_comparison}
\\begin{tabular}{|l|c|c|c|c|c|c|l|}
\\hline
\\textbf{Study} & \\textbf{Modality} & \\textbf{Train Acc.} & \\textbf{Test Acc.} & \\textbf{Macro F1} & \\textbf{Micro F1} & \\textbf{Precision} & \\textbf{Model} \\\\
\\hline
"""
    
    for row in rows:
        train_acc = row.get('Train Accuracy', '-')
        tex_content += f"{row['Study']} & {row['Modality']} & {train_acc} & {row['Test Accuracy']} & {row['Macro F1']} & {row['Micro F1']} & {row['Precision']} & {row['Model']} \\\\\n\\hline\n"
    
    tex_content += "\\end{tabular}\n\\end{table}\n"
    
    # Write to file
    with open(tex_file, "w") as f:
        f.write(tex_content)
    
    print(f"Updated {tex_file}")

def update_avd_table():
    """Update avd_table_latex.tex with data from updated_avd_metrics.csv"""
    csv_file = "updated_tables/updated_avd_metrics.csv"
    tex_file = "updated_tables/avd_table_latex.tex"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Skipping AVD table update.")
        return
    
    # Read CSV data
    rows = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Create LaTeX table content
    tex_content = """\\begin{table}[h]
\\centering
\\caption{Performance of best models for dimensional emotion (AVD) prediction using MAE and RMSE metrics}
\\label{tab:avd_prediction}
\\begin{tabular}{|l|c|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Modality} & \\textbf{Dimension} & \\textbf{Train RMSE} & \\textbf{Test RMSE} & \\textbf{Train MAE} & \\textbf{Test MAE} \\\\
\\hline
"""
    
    current_model = ""
    current_modality = ""
    
    for row in rows:
        if row['Model'] == current_model and row['Modality'] == current_modality:
            tex_content += f"  &   & {row['Dimension']} & {row['Train RMSE']} & {row['Test RMSE']} & {row['Train MAE']} & {row['Test MAE']} \\\\\n\\hline\n"
        else:
            current_model = row['Model']
            current_modality = row['Modality']
            tex_content += f"{row['Model']} & {row['Modality']} & {row['Dimension']} & {row['Train RMSE']} & {row['Test RMSE']} & {row['Train MAE']} & {row['Test MAE']} \\\\\n\\hline\n"
    
    tex_content += "\\end{tabular}\n\\end{table}\n"
    
    # Write to file
    with open(tex_file, "w") as f:
        f.write(tex_content)
    
    print(f"Updated {tex_file}")

def main():
    """Update all tables for the report"""
    print("Updating LaTeX tables from CSV data...")
    update_table3()
    update_table4()
    update_avd_table()
    print("Table updates complete.")

if __name__ == "__main__":
    main() 