#!/bin/bash
cd /Users/aarjavjain/Downloads/gbfao_project/results

# Create folders
mkdir -p "CEC-2014" "CEC-2017" "CEC-2020" "CEC-2022" "Engineering_Problems"

# Move CEC-2014 files
mv results_table.csv wilcoxon_results.csv wilcoxon_results_pvalues.csv rank_heatmap.pdf version_comparison.pdf "CEC-2014" 2>/dev/null
mv boxplots convergence "CEC-2014" 2>/dev/null

# Move CEC-2017 files
mv results_table_cec2017.csv wilcoxon_results_cec2017.csv wilcoxon_results_cec2017_pvalues.csv rank_heatmap_cec2017.pdf version_comparison_cec2017.pdf "CEC-2017" 2>/dev/null
mv boxplots_cec2017 convergence_cec2017 "CEC-2017" 2>/dev/null

# Move CEC-2020 files
mv cec2020_results.csv cec2020_wilcoxon.csv cec2020_wilcoxon_pvalues.csv rank_heatmap_cec2020.pdf version_comparison_cec2020.pdf "CEC-2020" 2>/dev/null
mv boxplots_cec2020 convergence_cec2020 "CEC-2020" 2>/dev/null

# Move CEC-2022 files
mv cec2022_results.csv cec2022_wilcoxon.csv cec2022_wilcoxon_pvalues.csv rank_heatmap_cec2022.pdf version_comparison_cec2022.pdf "CEC-2022" 2>/dev/null
mv boxplots_cec2022 convergence_cec2022 "CEC-2022" 2>/dev/null

# Move Engineering Problem files
mv engineering_results*.csv engineering_results*.pdf "Engineering_Problems" 2>/dev/null

echo "Organization complete."
