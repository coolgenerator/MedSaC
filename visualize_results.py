#!/usr/bin/env python3
"""
Visualization script for MedSaC evaluation results.
Generates comprehensive charts comparing different prompting methods.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_stats_files(stats_dir: str = "stats") -> Dict[str, Dict]:
    """Load all stats JSON files."""
    stats = {}
    for file in Path(stats_dir).glob("results_*.json"):
        # Extract method name from filename
        # e.g., results_VertexAI_gemini-2.5-flash_cot_rag_eval.json -> cot_rag
        parts = file.stem.split("_")
        method = "_".join(parts[3:-1])  # Get method name

        with open(file, "r") as f:
            stats[method] = json.load(f)
    return stats


def load_eval_files(eval_dir: str = "eval_output") -> Dict[str, List[Dict]]:
    """Load all evaluation JSON files."""
    evals = {}
    for file in Path(eval_dir).glob("*.json"):
        # Filename format: VertexAI_gemini-2.5-flash_cot_rag_eval.json
        parts = file.stem.split("_")
        method = "_".join(parts[2:-1])

        with open(file, "r") as f:
            evals[method] = json.load(f)
    return evals


def load_error_type_files(error_dir: str = "ErrorTypes") -> Dict[str, List[Dict]]:
    """Load all error type analysis JSON files."""
    errors = {}
    for file in Path(error_dir).glob("*_error_eval.json"):
        # Filename format: VertexAI_gemini-2.5-flash_cot_rag_error_eval.json
        parts = file.stem.split("_")
        # Find the method name (between model name and "error")
        # parts = ["VertexAI", "gemini-2.5-flash", "cot", "rag", "error", "eval"]
        error_idx = parts.index("error") if "error" in parts else -2
        method = "_".join(parts[2:error_idx])

        with open(file, "r") as f:
            errors[method] = json.load(f)
    return errors


def plot_overall_accuracy(stats: Dict[str, Dict], output_dir: str = "visualizations"):
    """Plot overall accuracy comparison between methods."""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(stats.keys())
    regex_acc = []
    llm_acc = []

    for method in methods:
        overall = stats[method].get("overall", {})
        regex_acc.append(overall.get("regular expression evaluation", {}).get("average", 0))
        llm_acc.append(overall.get("llm evaluation", {}).get("average", 0))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, regex_acc, width, label='Regex Evaluation', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, llm_acc, width, label='LLM Evaluation', color='#3498db', alpha=0.8)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/overall_accuracy.png")


def plot_category_accuracy(stats: Dict[str, Dict], output_dir: str = "visualizations"):
    """Plot accuracy by category for each method."""
    os.makedirs(output_dir, exist_ok=True)

    # Categories to plot (excluding meta categories)
    categories = ['date', 'diagnosis', 'dosage conversion', 'lab test', 'physical', 'risk', 'severity']
    methods = list(stats.keys())

    # Prepare data
    data = []
    for method in methods:
        for cat in categories:
            if cat in stats[method]:
                regex_acc = stats[method][cat].get("regular expression evaluation", {}).get("average", 0)
                llm_acc = stats[method][cat].get("llm evaluation", {}).get("average", 0)
                count = stats[method][cat].get("regular expression evaluation", {}).get("count", 0)
                data.append({
                    'Method': method.replace('_', ' ').title(),
                    'Category': cat.title(),
                    'Regex Accuracy': regex_acc,
                    'LLM Accuracy': llm_acc,
                    'Count': count
                })

    df = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(categories))
    width = 0.25
    multiplier = 0

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method.replace('_', ' ').title()]
        offset = width * multiplier
        bars = ax.bar(x + offset, method_data['Regex Accuracy'].values, width,
                     label=method.replace('_', ' ').title(), color=colors[i % len(colors)], alpha=0.8)
        multiplier += 1

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Regex Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Category and Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.title() for c in categories], rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/category_accuracy.png")


def plot_error_types_from_analysis(errors: Dict[str, List[Dict]], output_dir: str = "visualizations"):
    """
    Plot error type distribution from ErrorTypes analysis files.
    This matches the error types from the original MedRaC paper (Table 4).
    """
    os.makedirs(output_dir, exist_ok=True)

    if not errors:
        print("No error type analysis files found in ErrorTypes/")
        return

    # Error type columns (matching the paper's categories)
    error_columns = [
        "Formula Error",
        "Missing Variables",
        "Missing or Misused Demographic/Adjustment Coefficients",
        "Unit Conversion Error",
        "Arithmetic Errors",
        "Rounding / Precision Errors",
        "Incorrect Variable Extraction",
        "Clinical Misinterpretation (Rule-based Only)",
    ]

    # Short labels for display
    short_labels = [
        "Formula\nError",
        "Missing\nVariables",
        "Demographic\nCoeff.",
        "Unit\nConversion",
        "Arithmetic\nErrors",
        "Rounding/\nPrecision",
        "Variable\nExtraction",
        "Clinical\nMisinterp.",
    ]

    methods = list(errors.keys())

    # Count errors for each method
    error_counts = {}
    for method, records in errors.items():
        error_counts[method] = {}
        total = len(records)

        for col in error_columns:
            count = sum(1 for r in records if r.get(col) == "Yes")
            error_counts[method][col] = count

        error_counts[method]["total"] = total

    # --- Plot 1: Error counts table-style bar chart (like Table 4) ---
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(error_columns))
    width = 0.8 / len(methods)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, method in enumerate(methods):
        counts = [error_counts[method].get(col, 0) for col in error_columns]
        offset = width * (i - len(methods)/2 + 0.5)
        bars = ax.bar(x + offset, counts, width, label=method.replace('_', ' ').title(),
                     color=colors[i % len(colors)], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Error Count', fontsize=12)
    ax.set_title('Error Counts by Method and Error Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_type_counts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/error_type_counts.png")

    # --- Plot 2: Error rates (percentage) ---
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, method in enumerate(methods):
        total = error_counts[method]["total"]
        rates = [(error_counts[method].get(col, 0) / total * 100) if total > 0 else 0 for col in error_columns]
        offset = width * (i - len(methods)/2 + 0.5)
        bars = ax.bar(x + offset, rates, width, label=method.replace('_', ' ').title(),
                     color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Rate by Method and Error Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_type_rates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/error_type_rates.png")

    # --- Plot 3: Heatmap of error rates ---
    data = []
    for method in methods:
        total = error_counts[method]["total"]
        row = [(error_counts[method].get(col, 0) / total * 100) if total > 0 else 0 for col in error_columns]
        data.append(row)

    df = pd.DataFrame(data, index=[m.replace('_', ' ').title() for m in methods],
                     columns=short_labels)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn_r', center=25,
               vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Error Rate (%)'})

    ax.set_title('Error Type Heatmap (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_type_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/error_type_heatmap.png")

    # --- Print summary table (like Table 4 in paper) ---
    print("\n" + "=" * 100)
    print("Error Counts by Method and Error Type (Table 4 Style)")
    print("=" * 100)

    # Header
    header = f"{'Method':<20}"
    for label in short_labels:
        header += f"{label.replace(chr(10), ' '):<12}"
    print(header)
    print("-" * 100)

    # Data rows
    for method in methods:
        row = f"{method.replace('_', ' ').title():<20}"
        for col in error_columns:
            row += f"{error_counts[method].get(col, 0):<12}"
        print(row)

    return error_counts


def plot_question_type_comparison(stats: Dict[str, Dict], output_dir: str = "visualizations"):
    """Compare equation-based vs rule-based questions."""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(stats.keys())
    question_types = ['equation-based question', 'rule-based question']

    data = []
    for method in methods:
        for qt in question_types:
            if qt in stats[method]:
                regex_acc = stats[method][qt].get("regular expression evaluation", {}).get("average", 0)
                llm_acc = stats[method][qt].get("llm evaluation", {}).get("average", 0)
                data.append({
                    'Method': method.replace('_', ' ').title(),
                    'Question Type': qt.replace(' question', '').title(),
                    'Regex Accuracy': regex_acc,
                    'LLM Accuracy': llm_acc
                })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Regex accuracy
    pivot_regex = df.pivot(index='Method', columns='Question Type', values='Regex Accuracy')
    pivot_regex.plot(kind='bar', ax=axes[0], color=['#3498db', '#2ecc71'], alpha=0.8, rot=0)
    axes[0].set_title('Regex Evaluation Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 100)
    axes[0].legend(title='Question Type')

    # LLM accuracy
    pivot_llm = df.pivot(index='Method', columns='Question Type', values='LLM Accuracy')
    pivot_llm.plot(kind='bar', ax=axes[1], color=['#3498db', '#2ecc71'], alpha=0.8, rot=0)
    axes[1].set_title('LLM Evaluation Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend(title='Question Type')

    plt.suptitle('Equation-Based vs Rule-Based Questions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/question_type_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/question_type_comparison.png")


def plot_token_usage(stats: Dict[str, Dict], output_dir: str = "visualizations"):
    """Plot token usage comparison."""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(stats.keys())
    input_tokens = [stats[m].get("input_tokens_average", 0) for m in methods]
    output_tokens = [stats[m].get("output_tokens_average", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, input_tokens, width, label='Input Tokens', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, output_tokens, width, label='Output Tokens', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Average Tokens', fontsize=12)
    ax.set_title('Token Usage by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=10)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_usage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/token_usage.png")


def plot_detailed_eval_results(evals: Dict[str, List[Dict]], output_dir: str = "visualizations"):
    """Plot detailed evaluation results from eval_output files."""
    os.makedirs(output_dir, exist_ok=True)

    # Analyze correctness by calculator
    calculator_results = {}

    for method, records in evals.items():
        for record in records:
            calc_name = record.get("Calculator Name", "Unknown")
            result = record.get("Result", "Unknown")

            if calc_name not in calculator_results:
                calculator_results[calc_name] = {}
            if method not in calculator_results[calc_name]:
                calculator_results[calc_name][method] = {"Correct": 0, "Incorrect": 0}

            if result == "Correct":
                calculator_results[calc_name][method]["Correct"] += 1
            else:
                calculator_results[calc_name][method]["Incorrect"] += 1

    # Create dataframe
    data = []
    for calc_name, method_results in calculator_results.items():
        for method, counts in method_results.items():
            total = counts["Correct"] + counts["Incorrect"]
            accuracy = (counts["Correct"] / total * 100) if total > 0 else 0
            data.append({
                'Calculator': calc_name[:30] + '...' if len(calc_name) > 30 else calc_name,
                'Method': method.replace('_', ' ').title(),
                'Accuracy': accuracy,
                'Total': total
            })

    df = pd.DataFrame(data)

    if df.empty:
        print("No evaluation data to plot")
        return

    # Plot heatmap
    pivot = df.pivot(index='Calculator', columns='Method', values='Accuracy')

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})

    ax.set_title('Accuracy by Calculator and Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Calculator', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/calculator_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/calculator_heatmap.png")


def plot_llm_eval_error_counts(evals: Dict[str, List[Dict]], output_dir: str = "visualizations"):
    """
    Plot LLM evaluation error COUNTS (not rates) for each method.
    Shows exact numbers like Table 4 in the original paper.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not evals:
        print("No evaluation files found")
        return

    # Error types from LLM Evaluation
    error_types = ['formula', 'extracted_values', 'calculation', 'answer']
    labels = ['Formula\nError', 'Variable\nExtraction', 'Calculation\nError', 'Answer\nError']

    methods = list(evals.keys())

    # Count errors for each method
    error_counts = {}
    for method, records in evals.items():
        error_counts[method] = {et: 0 for et in error_types}
        error_counts[method]['total'] = len(records)
        error_counts[method]['incorrect'] = 0

        for record in records:
            if record.get("Result") == "Incorrect":
                error_counts[method]['incorrect'] += 1

            llm_eval = record.get("LLM Evaluation", {})
            for et in error_types:
                if llm_eval.get(et, {}).get("result") == "Incorrect":
                    error_counts[method][et] += 1

    # --- Plot: Error counts bar chart ---
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(error_types))
    width = 0.8 / len(methods)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for i, method in enumerate(methods):
        counts = [error_counts[method][et] for et in error_types]
        offset = width * (i - len(methods)/2 + 0.5)
        bars = ax.bar(x + offset, counts, width, label=method.replace('_', ' ').title(),
                     color=colors[i % len(colors)], alpha=0.85)

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                           fontsize=10, fontweight='bold')

    ax.set_xlabel('Error Type', fontsize=12)
    ax.set_ylabel('Error Count', fontsize=12)
    ax.set_title('LLM Evaluation Error Counts by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/llm_eval_error_counts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/llm_eval_error_counts.png")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print("LLM Evaluation Error Counts (Table 4 Style)")
    print("=" * 70)
    header = f"{'Method':<20} {'Total':<8} {'Incorrect':<10} {'Formula':<10} {'Extraction':<12} {'Calculation':<12} {'Answer':<10}"
    print(header)
    print("-" * 70)

    for method in methods:
        row = f"{method.replace('_', ' ').title():<20} "
        row += f"{error_counts[method]['total']:<8} "
        row += f"{error_counts[method]['incorrect']:<10} "
        row += f"{error_counts[method]['formula']:<10} "
        row += f"{error_counts[method]['extracted_values']:<12} "
        row += f"{error_counts[method]['calculation']:<12} "
        row += f"{error_counts[method]['answer']:<10}"
        print(row)

    return error_counts


def plot_summary_dashboard(stats: Dict[str, Dict], errors: Dict[str, List[Dict]], output_dir: str = "visualizations"):
    """Create a summary dashboard with key metrics."""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(stats.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall Accuracy (top-left)
    ax = axes[0, 0]
    regex_acc = [stats[m].get("overall", {}).get("regular expression evaluation", {}).get("average", 0) for m in methods]
    llm_acc = [stats[m].get("overall", {}).get("llm evaluation", {}).get("average", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, regex_acc, width, label='Regex', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, llm_acc, width, label='LLM', color='#3498db', alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods])
    ax.legend()
    ax.set_ylim(0, 100)

    # 2. Token Usage (top-right)
    ax = axes[0, 1]
    input_tokens = [stats[m].get("input_tokens_average", 0) for m in methods]
    output_tokens = [stats[m].get("output_tokens_average", 0) for m in methods]

    ax.bar(x - width/2, input_tokens, width, label='Input', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, output_tokens, width, label='Output', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Tokens')
    ax.set_title('Average Token Usage', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods])
    ax.legend()

    # 3. Question Type Performance (bottom-left)
    ax = axes[1, 0]
    eq_acc = [stats[m].get("equation-based question", {}).get("regular expression evaluation", {}).get("average", 0) for m in methods]
    rule_acc = [stats[m].get("rule-based question", {}).get("regular expression evaluation", {}).get("average", 0) for m in methods]

    ax.bar(x - width/2, eq_acc, width, label='Equation-based', color='#9b59b6', alpha=0.8)
    ax.bar(x + width/2, rule_acc, width, label='Rule-based', color='#f39c12', alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Question Type Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods])
    ax.legend()
    ax.set_ylim(0, 100)

    # 4. Error Types from ErrorTypes analysis (bottom-right)
    ax = axes[1, 1]

    if errors:
        # Key error types to show
        error_cols = [
            "Formula Error",
            "Arithmetic Errors",
            "Incorrect Variable Extraction",
            "Missing Variables",
        ]
        short_labels = ['Formula', 'Arithmetic', 'Extraction', 'Missing Var']
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db']

        bar_width = 0.2
        for i, method in enumerate(methods):
            if method in errors:
                total = len(errors[method])
                rates = []
                for col in error_cols:
                    count = sum(1 for r in errors[method] if r.get(col) == "Yes")
                    rates.append((count / total * 100) if total > 0 else 0)

                offset = bar_width * (i - len(methods)/2 + 0.5)
                ax.bar(np.arange(len(error_cols)) + offset, rates, bar_width,
                      label=method.replace('_', ' ').title(), alpha=0.8)

        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Key Error Types', fontweight='bold')
        ax.set_xticks(np.arange(len(error_cols)))
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'No error analysis data\n(Run error_type_pipeline first)',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Key Error Types', fontweight='bold')

    plt.suptitle('MedSaC Evaluation Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/summary_dashboard.png")


def generate_report(stats: Dict[str, Dict], errors: Dict[str, List[Dict]], output_dir: str = "visualizations"):
    """Generate a text summary report."""
    os.makedirs(output_dir, exist_ok=True)

    report_lines = [
        "=" * 80,
        "MedSaC Evaluation Report",
        "=" * 80,
        ""
    ]

    for method in stats.keys():
        overall = stats[method].get("overall", {})
        regex_acc = overall.get("regular expression evaluation", {}).get("average", 0)
        llm_acc = overall.get("llm evaluation", {}).get("average", 0)
        count = overall.get("regular expression evaluation", {}).get("count", 0)
        input_tokens = stats[method].get("input_tokens_average", 0)
        output_tokens = stats[method].get("output_tokens_average", 0)

        report_lines.extend([
            f"Method: {method.upper()}",
            "-" * 40,
            f"  Total Questions: {count}",
            f"  Regex Accuracy: {regex_acc:.2f}%",
            f"  LLM Accuracy: {llm_acc:.2f}%",
            f"  Avg Input Tokens: {input_tokens}",
            f"  Avg Output Tokens: {output_tokens}",
            ""
        ])

        # Error breakdown from ErrorTypes
        if method in errors:
            error_cols = [
                "Formula Error",
                "Missing Variables",
                "Missing or Misused Demographic/Adjustment Coefficients",
                "Unit Conversion Error",
                "Arithmetic Errors",
                "Rounding / Precision Errors",
                "Incorrect Variable Extraction",
                "Clinical Misinterpretation (Rule-based Only)",
            ]

            total = len(errors[method])
            report_lines.append("  Error Type Breakdown:")

            for col in error_cols:
                count = sum(1 for r in errors[method] if r.get(col) == "Yes")
                rate = (count / total * 100) if total > 0 else 0
                report_lines.append(f"    - {col}: {count} ({rate:.1f}%)")

            report_lines.extend(["", ""])

    report = "\n".join(report_lines)

    with open(f"{output_dir}/report.txt", "w") as f:
        f.write(report)

    print(f"Saved: {output_dir}/report.txt")
    print("\n" + report)


def main():
    """Main function to generate all visualizations."""
    print("Loading data...")
    stats = load_stats_files("stats")
    evals = load_eval_files("eval_output")
    errors = load_error_type_files("ErrorTypes")

    if not stats:
        print("No stats files found in 'stats/' directory")
        return

    print(f"Found {len(stats)} methods in stats: {list(stats.keys())}")
    print(f"Found {len(evals)} methods in eval_output: {list(evals.keys())}")
    print(f"Found {len(errors)} methods in ErrorTypes: {list(errors.keys())}")
    print("\nGenerating visualizations...")

    output_dir = "visualizations"

    # Generate all plots
    plot_overall_accuracy(stats, output_dir)
    plot_category_accuracy(stats, output_dir)
    plot_question_type_comparison(stats, output_dir)
    plot_token_usage(stats, output_dir)

    if evals:
        plot_detailed_eval_results(evals, output_dir)
        # LLM Evaluation error counts (like Table 4 in paper)
        plot_llm_eval_error_counts(evals, output_dir)

    # Error type analysis from ErrorTypes folder (8 detailed error types)
    if errors:
        plot_error_types_from_analysis(errors, output_dir)

    plot_summary_dashboard(stats, errors, output_dir)
    generate_report(stats, errors, output_dir)

    print(f"\nAll visualizations saved to '{output_dir}/' directory")


if __name__ == "__main__":
    main()
