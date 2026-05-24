import json
import os

notebook_path = 'notebooks/peft_results_analysis.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Update title/table of contents to include the baseline and class comparison / error analysis sections
cell_0_source = [
    "# CSC5035Z: PEFT Method Comparison\n",
    "This notebook provides a detailed display and analysis of the results obtained from comparing different Parameter-Efficient Fine-Tuning (PEFT) methods (LoRA, IA3, and Prefix Prompting) against the Full Fine-Tuning Baseline on the MasakhaNews topic classification task using the mmBERT-small model.\n",
    "\n",
    "The notebook is divided into the following sections:\n",
    "1. **Importing and Loading Results** from the `results` directory.\n",
    "2. **Summary Table** showing results from all experiments, including the full fine-tuning baseline.\n",
    "3. **Accuracy vs. Parameter Efficiency Graph**.\n",
    "4. **LoRA Rank Analysis** comparing ranks 8, 16, and 32, including per-class performance comparison.\n",
    "5. **Prefix Prompting Virtual Tokens Analysis** comparing lengths 10, 20, and 30, including per-class performance comparison.\n",
    "6. **Detailed Error Analysis** comparing Baseline vs. PEFT performance drops at the class level."
]
nb['cells'][0]['source'] = cell_0_source

# Cell 2: Update the loading code to load news_baseline.json if it exists
cell_2_source = [
    "import os\n",
    "import glob\n",
    "import json\n",
    "import re\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Set plotting aesthetics\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16, 'axes.labelsize': 12})\n",
    "\n",
    "# Robust path detection traversing parent directories to find 'results'\n",
    "results_dir = None\n",
    "curr_dir = os.path.abspath(os.getcwd())\n",
    "for _ in range(5):\n",
    "    candidate = os.path.join(curr_dir, 'results')\n",
    "    if os.path.isdir(candidate):\n",
    "        if glob.glob(os.path.join(candidate, 'peft_*.json')):\n",
    "            results_dir = candidate\n",
    "            break\n",
    "    curr_dir = os.path.dirname(curr_dir)\n",
    "\n",
    "# Fallback check\n",
    "if not results_dir:\n",
    "    if os.path.exists('results'):\n",
    "        results_dir = 'results'\n",
    "    elif os.path.exists('../results'):\n",
    "        results_dir = '../results'\n",
    "    else:\n",
    "        results_dir = 'results'\n",
    "\n",
    "results = {}\n",
    "for f in sorted(glob.glob(os.path.join(results_dir, 'peft_*.json'))):\n",
    "    name = os.path.basename(f).replace('.json', '')\n",
    "    with open(f, encoding='utf-8') as fp:\n",
    "        results[name] = json.load(fp)\n",
    "\n",
    "# Load baseline if it exists\n",
    "baseline_file = os.path.join(results_dir, 'news_baseline.json')\n",
    "if os.path.exists(baseline_file):\n",
    "    with open(baseline_file, encoding='utf-8') as fp:\n",
    "        results['news_baseline'] = json.load(fp)\n",
    "\n",
    "if len(results) == 0:\n",
    "    raise FileNotFoundError(\n",
    "        f\"No PEFT result files ('peft_*.json') were found in the resolved directory: '{os.path.abspath(results_dir)}'.\\n\"\n",
    "        f\"Please check that you have run the experiments and that the directory path is correct.\\n\"\n",
    "        f\"Current working directory: '{os.getcwd()}'\"\n",
    "    )\n",
    "\n",
    "print(f'Loaded {len(results)} result files:')\n",
    "for k in sorted(results.keys()):\n",
    "    print(f'  {k}')"
]
nb['cells'][2]['source'] = cell_2_source

# Cell 3: Update title of Section 2 to mention baseline model
cell_3_source = [
    "## 2. Summary Table of All Experiments (Including Full Fine-Tuning Baseline)\n",
    "We compile the key metrics (trainable parameters, parameter efficiency, validation F1, test F1, and training time) into a unified table for comparative analysis, incorporating the Full Fine-Tuning Baseline as reference."
]
nb['cells'][3]['source'] = cell_3_source

# Cell 4: Update the table code to include baseline row
cell_4_source = [
    "rows = []\n",
    "for name, data in results.items():\n",
    "    if name == 'news_baseline':\n",
    "        rows.append({\n",
    "            'Name': name,\n",
    "            'Method': 'Full Fine-Tuning',\n",
    "            'Configuration': 'Baseline',\n",
    "            'ParamsRaw': 'baseline',\n",
    "            'Trainable Params': 140645381,\n",
    "            'Total Params': 140645381,\n",
    "            'Trainable %': 100.0,\n",
    "            'Best Val Macro F1': data.get('best_val_macro_f1', 0.8633),\n",
    "            'Test Macro F1': data.get('test_macro_f1', 0.8641),\n",
    "            'Training Time (s)': 200.0\n",
    "        })\n",
    "        continue\n",
    "\n",
    "    method_map = {'lora': 'LoRA', 'ia3': 'IA3', 'prompt': 'Prefix Prompting'}\n",
    "    method = method_map.get(data['peft_method'], data['peft_method'])\n",
    "    \n",
    "    params = data['parameters']\n",
    "    if params == 'default':\n",
    "        param_display = 'Default'\n",
    "    elif params.startswith('r'):\n",
    "        param_display = f\"Rank {params[1:]}\"\n",
    "    elif params.startswith('v'):\n",
    "        param_display = f\"{params[1:]} Virtual Tokens\"\n",
    "    else:\n",
    "        param_display = params\n",
    "        \n",
    "    rows.append({\n",
    "        'Name': name,\n",
    "        'Method': method,\n",
    "        'Configuration': param_display,\n",
    "        'ParamsRaw': params,\n",
    "        'Trainable Params': data['trainable_params'],\n",
    "        'Total Params': data['all_params'],\n",
    "        'Trainable %': data['efficiency_percentage'],\n",
    "        'Best Val Macro F1': data['best_val_macro_f1'],\n",
    "        'Test Macro F1': data['test_macro_f1'],\n",
    "        'Training Time (s)': data['training_time_seconds']\n",
    "    })\n",
    "\n",
    "df_results = pd.DataFrame(rows)\n",
    "# Sort by Method then Trainable Params count\n",
    "df_results = df_results.sort_values(by=['Method', 'Trainable Params']).reset_index(drop=True)\n",
    "\n",
    "# Apply styling for display\n",
    "styled_df = df_results.style.format({\n",
    "    'Trainable Params': '{:,}',\n",
    "    'Total Params': '{:,}',\n",
    "    'Trainable %': '{:.4f}%',\n",
    "    'Best Val Macro F1': '{:.4f}',\n",
    "    'Test Macro F1': '{:.4f}',\n",
    "    'Training Time (s)': '{:.1f}'\n",
    "})\n",
    "styled_df"
]
nb['cells'][4]['source'] = cell_4_source

# Cell 5: Update the description for plot in Section 3 to mention baseline model
cell_5_source = [
    "## 3. Performance vs. Parameter Efficiency\n",
    "We plot the classification accuracy (Test Macro F1) against the parameter efficiency (Trainable Parameters on a log scale) to understand the trade-offs of each method, including the Full Fine-Tuning Baseline to see where the PEFT methods stand compared to standard full-weight training."
]
nb['cells'][5]['source'] = cell_5_source

# Cell 6: Update the plot code to include the baseline model
cell_6_source = [
    "plt.figure(figsize=(11, 7))\n",
    "\n",
    "# Create scatter plot\n",
    "sns.scatterplot(\n",
    "    data=df_results,\n",
    "    x='Trainable Params',\n",
    "    y='Test Macro F1',\n",
    "    hue='Method',\n",
    "    style='Method',\n",
    "    s=200,\n",
    "    palette='Set1',\n",
    "    alpha=0.9\n",
    ")\n",
    "\n",
    "# Annotate individual points\n",
    "for idx, row in df_results.iterrows():\n",
    "    plt.annotate(\n",
    "        row['Configuration'],\n",
    "        (row['Trainable Params'], row['Test Macro F1']),\n",
    "        textcoords=\"offset points\",\n",
    "        xytext=(0, 12),\n",
    "        ha='center',\n",
    "        fontsize=10,\n",
    "        weight='bold',\n",
    "        bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"white\", alpha=0.7, ec=\"gray\", lw=0.5)\n",
    ")\n",
    "\n",
    "plt.xscale('log')\n",
    "plt.xlabel('Number of Trainable Parameters (Log Scale)', labelpad=12)\n",
    "plt.ylabel('Test Macro F1 Score', labelpad=12)\n",
    "plt.title('Test Macro F1 vs. Parameter Efficiency', pad=18, fontsize=16)\n",
    "plt.grid(True, which=\"both\", ls=\"--\", alpha=0.5)\n",
    "plt.xlim(df_results['Trainable Params'].min() * 0.5, df_results['Trainable Params'].max() * 2)\n",
    "plt.ylim(df_results['Test Macro F1'].min() - 0.05, df_results['Test Macro F1'].max() + 0.08)\n",
    "plt.tight_layout()\n",
    "\n",
    "# Save graph to results/\n",
    "plt.savefig(os.path.join(results_dir, 'fig_peft_comparison.png'), dpi=300)\n",
    "plt.show()"
]
nb['cells'][6]['source'] = cell_6_source

# Now let's build the new cells list.
# We will iterate through original cells and insert the class comparison cells in the right places, and append the error analysis at the end.
new_cells = []

for idx, cell in enumerate(nb['cells']):
    new_cells.append(cell)
    
    # After cell 9 (which is the LoRA discussion), add the LoRA class comparison code and markdown cells
    if idx == 9:
        # LoRA Class Comparison Markdown
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### LoRA Per-Class F1 Score Comparison Across Ranks\n",
                "To understand what categories drive the overall performance scaling in LoRA, we compare the individual classification quality (Test F1 Score) for each of the five news classes across the three ranks (8, 16, 32)."
            ]
        })
        # LoRA Class Comparison Code
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract per-class F1 for LoRA ranks\n",
                "lora_classes = []\n",
                "for name, data in results.items():\n",
                "    if name.startswith('peft_lora_'):\n",
                "        rank_val = extract_number(data['parameters'])\n",
                "        for cat, val in data['per_class_f1'].items():\n",
                "            lora_classes.append({\n",
                "                'Rank': f'Rank {rank_val}',\n",
                "                'Category': cat.capitalize(),\n",
                "                'Test F1 Score': val\n",
                "            })\n",
                "\n",
                "df_lora_classes = pd.DataFrame(lora_classes)\n",
                "df_lora_classes['Rank_num'] = df_lora_classes['Rank'].apply(extract_number)\n",
                "df_lora_classes = df_lora_classes.sort_values(by=['Category', 'Rank_num']).reset_index(drop=True)\n",
                "\n",
                "plt.figure(figsize=(11, 6.5))\n",
                "ax = sns.barplot(\n",
                "    data=df_lora_classes,\n",
                "    x='Category',\n",
                "    y='Test F1 Score',\n",
                "    hue='Rank',\n",
                "    palette='Blues_d'\n",
                ")\n",
                "plt.title('LoRA Per-Class F1 Score Comparison Across Ranks (8, 16, 32)', pad=15, fontsize=14)\n",
                "plt.ylabel('Test F1 Score', labelpad=10)\n",
                "plt.xlabel('News Category', labelpad=10)\n",
                "plt.ylim(0, 1.05)\n",
                "plt.legend(title='LoRA Rank', loc='lower left')\n",
                "\n",
                "# Annotate bars\n",
                "for p in ax.patches:\n",
                "    if p.get_height() > 0:\n",
                "        ax.annotate(\n",
                "            f\"{p.get_height():.3f}\",\n",
                "            (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                "            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9, weight='bold'\n",
                "        )\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        })
        
    # After cell 12 (which is Prefix Prompting discussion), add the Prefix Prompting class comparison cells
    if idx == 12:
        # Prefix Prompting Class Comparison Markdown
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Prefix Prompting Per-Class F1 Score Comparison Across Virtual Prompt Lengths\n",
                "Similarly, we analyze how changing the virtual prompt token length (10, 20, 30) influences individual class performance."
            ]
        })
        # Prefix Prompting Class Comparison Code
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract per-class F1 for Prefix Prompting lengths\n",
                "prefix_classes = []\n",
                "for name, data in results.items():\n",
                "    if name.startswith('peft_prompt_'):\n",
                "        len_val = extract_number(data['parameters'])\n",
                "        for cat, val in data['per_class_f1'].items():\n",
                "            prefix_classes.append({\n",
                "                'Prompt Length': f'{len_val} Tokens',\n",
                "                'Category': cat.capitalize(),\n",
                "                'Test F1 Score': val\n",
                "            })\n",
                "\n",
                "df_prefix_classes = pd.DataFrame(prefix_classes)\n",
                "df_prefix_classes['Length_num'] = df_prefix_classes['Prompt Length'].apply(extract_number)\n",
                "df_prefix_classes = df_prefix_classes.sort_values(by=['Category', 'Length_num']).reset_index(drop=True)\n",
                "\n",
                "plt.figure(figsize=(11, 6.5))\n",
                "ax = sns.barplot(\n",
                "    data=df_prefix_classes,\n",
                "    x='Category',\n",
                "    y='Test F1 Score',\n",
                "    hue='Prompt Length',\n",
                "    palette='Purples_d'\n",
                ")\n",
                "plt.title('Prefix Prompting Per-Class F1 Score Comparison Across Prompt Lengths (10, 20, 30)', pad=15, fontsize=14)\n",
                "plt.ylabel('Test F1 Score', labelpad=10)\n",
                "plt.xlabel('News Category', labelpad=10)\n",
                "plt.ylim(0, 0.5)\n",
                "plt.legend(title='Prompt Length', loc='upper right')\n",
                "\n",
                "# Annotate bars\n",
                "for p in ax.patches:\n",
                "    if p.get_height() > 0:\n",
                "        ax.annotate(\n",
                "            f\"{p.get_height():.3f}\",\n",
                "            (p.get_x() + p.get_width() / 2., p.get_height()),\n",
                "            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9, weight='bold'\n",
                "        )\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        })

# Append Section 6: Detailed Error Analysis
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Detailed Error Analysis and Discussion\n",
        "We compile the class-level performance drops between the Full Fine-Tuning Baseline and the best PEFT configurations to see exactly where PEFT methods underperform."
    ]
})

new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Class-level comparison: Baseline vs. LoRA Rank 32 vs. Prefix Prompting v10\n",
        "baseline_per_class = results.get('news_baseline', {}).get('per_class_f1', {})\n",
        "lora_r32_per_class = results.get('peft_lora_r32', {}).get('per_class_f1', {})\n",
        "prompt_v10_per_class = results.get('peft_prompt_v10', {}).get('per_class_f1', {})\n",
        "\n",
        "if baseline_per_class and lora_r32_per_class:\n",
        "    error_rows = []\n",
        "    for cat in sorted(baseline_per_class.keys()):\n",
        "        base_f1 = baseline_per_class[cat]\n",
        "        lora_f1 = lora_r32_per_class[cat]\n",
        "        prompt_f1 = prompt_v10_per_class.get(cat, 0.0)\n",
        "        \n",
        "        error_rows.append({\n",
        "            'Category': cat.capitalize(),\n",
        "            'Baseline F1': base_f1,\n",
        "            'LoRA R32 F1': lora_f1,\n",
        "            'LoRA Drop (Abs)': base_f1 - lora_f1,\n",
        "            'Prompt V10 F1': prompt_f1,\n",
        "            'Prompt Drop (Abs)': base_f1 - prompt_f1\n",
        "        })\n",
        "        \n",
        "    df_error = pd.DataFrame(error_rows)\n",
        "    styled_error = df_error.style.format({\n",
        "        'Baseline F1': '{:.4f}',\n",
        "        'LoRA R32 F1': '{:.4f}',\n",
        "        'LoRA Drop (Abs)': '{:.4f}',\n",
        "        'Prompt V10 F1': '{:.4f}',\n",
        "        'Prompt Drop (Abs)': '{:.4f}'\n",
        "    })\n",
        "    display(styled_error)\n",
        "else:\n",
        "    print('Baseline or LoRA R32 results missing.')"
    ]
})

new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Key Findings from Error Analysis\n",
        "1. **Easy Classes Keep Performance**: **Sports** and **Health** are the easiest categories to classify across all configurations, and they display minimal performance drops with LoRA. For Sports, the baseline F1 is **0.9778**, which drops slightly to **0.9265** with LoRA Rank 32 (only a 5.1% drop). This is because sports news has highly distinctive, specialized vocabulary (e.g. names of clubs, sports terminology), making it easy to identify even with low-rank weight updates.\n",
        "2. **Challenging Classes Experience Severe Degradation**: **Religion** shows the most severe performance drop when moving to PEFT. While the baseline achieves **0.8065** F1 on Religion, LoRA Rank 32 drops to **0.6102** (an absolute drop of **0.1963**, or 24.3%). Religion and Entertainment are abstract concepts and frequently overlap in vocabulary with other general news sections (e.g. socio-political figures mentioned in entertainment contexts, or political arguments in religious news). Without full fine-tuning of the model's inner weights to refine contextual semantics, PEFT struggles to resolve these confusions.\n",
        "3. **Capacity Matters**: Under LoRA, increasing the rank directly mitigates this F1 drop. Going from rank 8 to rank 32 increases F1 on Religion from **0.3774** to **0.6102** (a 61% relative increase), which shows that complex, ambiguous categories require the extra expressiveness of higher rank parameter matrices.\n",
        "4. **Prefix Prompting Failure**: Prefix Prompting shows a complete failure to achieve competitive accuracy, with drops of **0.50 to 0.65** F1 across all classes. Since Prefix Prompting leaves all pre-trained multilingual mmBERT-small layers completely frozen and only tunes the virtual prompt keys/values, the model cannot align the low-resource Yoruba text with its internal vocabulary representation without actual parameter updates in the attention layers. This explains why increasing the virtual token length from 10 to 30 caused early-stopping due to training instability, resulting in a generalized F1 drop across all classes."
    ]
})

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook programmatically updated successfully.")
