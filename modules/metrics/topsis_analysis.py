import logging
import numpy as np
from modules.lib.constants import RMSE, MRMSE, HITRATE, CSI, F2SCORE, F3SCORE, INFERENCE_TIMES, INFERENCE_MEMORY_USAGE, FLOPS, PARAMS, OUTPUT_DIR
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

weight_profiles = [{RMSE: 0.1, MRMSE: 0.1, HITRATE: 0.1, CSI: 0.1, F2SCORE: 0.1, F3SCORE: 0.1, INFERENCE_TIMES: 0.1, INFERENCE_MEMORY_USAGE: 0.1, FLOPS: 0.1, PARAMS: 0.1},
                       {RMSE: 0.99/6, MRMSE: 0.99/6, HITRATE: 0.99/6, CSI: 0.99/6, F2SCORE: 0.99/6, F3SCORE: 0.99/6, INFERENCE_TIMES: 0.0025, INFERENCE_MEMORY_USAGE: 0.0025, FLOPS: 0.0025, PARAMS: 0.0025},
                       {RMSE: 0.01/6, MRMSE: 0.01/6, HITRATE: 0.01/6, CSI: 0.01/6, F2SCORE: 0.01/6, F3SCORE: 0.01/6, INFERENCE_TIMES: 0.2475, INFERENCE_MEMORY_USAGE: 0.2475, FLOPS: 0.2475, PARAMS: 0.2475},
                       {RMSE: 0.99/7, MRMSE: 0.99/7, HITRATE: 0.99/7, CSI: 0.99/7, F2SCORE: 0.99/7, F3SCORE: 0.99/7, INFERENCE_TIMES: 0.0, INFERENCE_MEMORY_USAGE: 0.01/3, FLOPS: 0.01/3, PARAMS: 0.01/3},]


model_colors = {
    "Tier-1": "#0173B2", 
    "Tier-2": "#DE8F05",   
    "Tier-3": "#029E73",       
    "Tier-4": "#D55E00",      
}

def create_radar_chart(models, normalised_matrix, output_dir):
    import matplotlib.pyplot as plt
    from math import pi

    ordered_models = ["Tier-1", "Tier-2", "Tier-3", "Tier-4"]
    categories = list(normalised_matrix.keys())
    N = len(categories)
    
    #Add 0.05 if any value is zero to avoid issues in plotting
    for cat in categories:
        for i in range(len(normalised_matrix[cat])):
            if normalised_matrix[cat][i] == 0:
                normalised_matrix[cat][i] += 0.05

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Create figure with higher DPI
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = plt.subplot(111, polar=True)
    
    # Customize grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_facecolor('#f8f9fa')

    for model in ordered_models:
        idx = list(models).index(model)
        values = [normalised_matrix[cat][idx] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=model, 
                color=model_colors.get(model, None), marker='o', markersize=6)
        ax.fill(angles, values, alpha=0.15, color=model_colors.get(model, None))

    # Format category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight='bold')
    
    # Improve radial axis
    ax.set_rlabel_position(90)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
               color="grey", size=9)
    plt.ylim(0, 1)

    # Better legend positioning
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), 
               frameon=True, shadow=True, fontsize=18, ncol=1)
    
    # plt.title("Model Performance Radar Chart", size=18, weight='bold', pad=20)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "model_performance_radar_chart.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Radar chart saved to {output_file}")

def topsis(models, model_metrics):
    nomralised_matrix = normalise_metrics(models, model_metrics)
    output_dir = os.path.join(OUTPUT_DIR,"topsis_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #create a radar chart for models based on normalised metrics
    create_radar_chart(models, nomralised_matrix, output_dir)
        
    model_scores = {}
    for idx in range(len(weight_profiles)):
        settings_index = idx
        weighted_matrix = calculate_weighted_matrix(nomralised_matrix, settings_index)
        a_scores_dict = find_best_and_worst(weighted_matrix)
        distances_matrix = find_distances(models, weighted_matrix, a_scores_dict)
        scores_matrix =  compute_topsis_scores(distances_matrix)
        model_scores[settings_index] = scores_matrix
        
        # save_all_matrices
        weighted_df = pd.DataFrame(weighted_matrix)
        weighted_df.to_csv(os.path.join(output_dir, f"weighted_matrix_settings_{settings_index}.csv"), index=False)
        nomralised_df = pd.DataFrame(nomralised_matrix)
        nomralised_df.to_csv(os.path.join(output_dir, f"normalised_matrix_settings_{settings_index}.csv"), index=False)
        distances_df = pd.DataFrame(distances_matrix)
        distances_df.to_csv(os.path.join(output_dir, f"distances_matrix_settings_{settings_index}.csv"), index=False)
        a_scores_df = pd.DataFrame.from_dict(a_scores_dict, orient='index')
        a_scores_df.to_csv(os.path.join(output_dir, f"a_scores_settings_{settings_index}.csv"))
        
        topsis_df = pd.DataFrame(scores_matrix)
        topsis_df.to_csv(os.path.join(output_dir, f"topsis_scores_settings_{settings_index}.csv"), index=False)
        
    # Write to csv
    output_file = os.path.join(output_dir, "topsis_scores.csv")
    all_scores = []
    for setting_index, scores in model_scores.items():
        for score_entry in scores:
            all_scores.append({
                'settings_index': setting_index,
                'model': score_entry['model'],
                'topsis_score': score_entry['topsis_score']
            })
    df = pd.DataFrame(all_scores)
    df.to_csv(output_file, index=False)
    return model_scores
    
def normalise_metrics(models, metrics_dict):
    
    metrics_list = [RMSE, MRMSE, HITRATE, CSI, F2SCORE, F3SCORE, INFERENCE_TIMES, INFERENCE_MEMORY_USAGE, FLOPS, PARAMS]
    cost_metrics = [RMSE, MRMSE, INFERENCE_TIMES, INFERENCE_MEMORY_USAGE, FLOPS, PARAMS] # metrics where lower is better
    logged_metrics = [RMSE, MRMSE, FLOPS, PARAMS]
    
    normalsied_matrix = {}
    
    def ensure_numeric(values):
        result = []
        for val in values:
            if isinstance(val, str):
                try:
                    if val.startswith('{'):
                        # Handle memory usage dictionaries
                        mem_dict = eval(val)
                        result.append(mem_dict.get('max_cuda_memory', 0) / (1024**3))
                    else:
                        result.append(float(val))
                except:
                    logger.warning(f"Could not convert value: {val} to numeric, using 1.0")
                    result.append(1.0)  # Use 1.0 instead of 0 to avoid division by zero
            else:
                result.append(float(val) if val is not None else 1.0)
        return result
        
    def normalise(values, metric):
        if metric in logged_metrics:
            values = [np.log(v) for v in values]
            
        max_val = max(values)
        min_val = min(values)
        
        if max_val == min_val:
            return [1.0] * len(values)
        
        if metric in cost_metrics:
            values = [(max_val - v) / (max_val - min_val) for v in values]
        else: 
            values = [(v - min_val) / (max_val - min_val) for v in values]
        return values
        
    for metric in metrics_list:
        numeric_metric = ensure_numeric(metrics_dict[metric])
        normalsied_matrix[metric] = normalise(numeric_metric, metric)
    return normalsied_matrix
            
def calculate_weighted_matrix(normalised_matrix, settings_index):
    weights = weight_profiles[settings_index]
    weighted_matrix = {}
    
    for metric, values in normalised_matrix.items():
        weight = weights.get(metric, 0)
        weighted_values = [v * weight for v in values]
        weighted_matrix[metric] = weighted_values
    
    return weighted_matrix


def find_best_and_worst(weighted_matrix):
    best_and_worst = {}
    for metric, values in weighted_matrix.items():
        best_and_worst[metric] = {
            'a_plus': max(values),
            'a_minus': min(values)
        }
    return best_and_worst
            
def find_distances(models, weighted_matrix, a_scores_dict):
    distances = []
    num_models = len(models)
    
    for i in range(num_models):
        d_plus = 0
        d_minus = 0
        for metric, values in weighted_matrix.items():
            a_plus = a_scores_dict[metric]['a_plus']
            a_minus = a_scores_dict[metric]['a_minus']
            d_plus += (values[i] - a_plus) ** 2
            d_minus += (values[i] - a_minus) ** 2
        d_plus = np.sqrt(d_plus)
        d_minus = np.sqrt(d_minus)
        distances.append({'model': models[i], 'd_plus': d_plus, 'd_minus': d_minus})
    return distances

def compute_topsis_scores(distances_matrix):
    scores = []
    for entry in distances_matrix:
        d_plus = entry['d_plus']
        d_minus = entry['d_minus']
        if (d_plus + d_minus) == 0:
            score = 0
        else:
            score = d_minus / (d_plus + d_minus)
        scores.append({'model': entry['model'], 'topsis_score': score})
    return scores