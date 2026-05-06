import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results():
    results_dir = '../results'
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist.")
        return

    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    if not json_files:
        print("No JSON files found in results directory.")
        return

    single_needle_records = []
    multi_needle_records = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'context_length' in data and 'depth_percent' in data:
                single_needle_records.append(data)
            elif 'total_files' in data or 'test_number' in data:
                multi_needle_records.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Visualize single needle results (Heatmap)
    if single_needle_records:
        df_single = pd.DataFrame(single_needle_records)
        df_single['depth_percent'] = df_single['depth_percent'].round(1)
        df_single['context_length'] = df_single['context_length'].astype(int)
        df_single['score'] = df_single['score'].astype(float)

        pivot_table = df_single.pivot_table(
            index='depth_percent', 
            columns='context_length', 
            values='score', 
            aggfunc='mean'
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Score'})
        plt.title('Single Needle in a Haystack - Score Heatmap')
        plt.xlabel('Context Length (Tokens)')
        plt.ylabel('Depth Percent (%)')
        plt.tight_layout()
        single_output = 'single_needle_heatmap.png'
        plt.savefig(single_output)
        print(f"Saved single needle heatmap to {single_output}")
        plt.close()

    # Visualize multi needle results
    if multi_needle_records:
        df_multi = pd.DataFrame(multi_needle_records)
        
        # Plot 1: Scores across tests
        plt.figure(figsize=(10, 6))
        if 'test_number' in df_multi.columns and not df_multi['test_number'].isnull().all():
            sns.barplot(data=df_multi, x='test_number', y='score', palette='viridis')
            plt.xlabel('Test Number')
        else:
            df_multi['Index'] = range(1, len(df_multi) + 1)
            sns.barplot(data=df_multi, x='Index', y='score', palette='viridis')
            plt.xlabel('Test Index')
            
        plt.title('Multi Needle Tests - Scores')
        plt.ylabel('Score')
        plt.ylim(0, 10)  # Assuming max score is usually around 10
        plt.tight_layout()
        multi_output = 'multi_needle_scores.png'
        plt.savefig(multi_output)
        print(f"Saved multi needle scores bar chart to {multi_output}")
        plt.close()

        # Plot 2: Bad case attribution (Pie chart of Scores)
        plt.figure(figsize=(8, 8))
        def categorize_score(s):
            if s >= 8: return 'Perfect / Good (>=8)'
            elif s >= 4: return 'Partial (4-7)'
            else: return 'Fail (<4)'
            
        df_multi['Outcome'] = df_multi['score'].apply(categorize_score)
        outcome_counts = df_multi['Outcome'].value_counts()
        plt.pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        plt.title('Multi Needle Tests - Performance Breakdown')
        plt.tight_layout()
        pie_output = 'multi_needle_performance_pie.png'
        plt.savefig(pie_output)
        print(f"Saved multi needle performance pie chart to {pie_output}")
        plt.close()

if __name__ == '__main__':
    visualize_results()
