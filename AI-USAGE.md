# AI_USAGE.md — Use of AI Tools

## 1) Tools used
- **ChatGPT (OpenAI)** — primary writing assistant for LaTeX polishing, structure suggestions, methodology review, and code organization recommendations.
- **Claude (Anthropic)** — secondary tool for code review suggestions, LaTeX formatting, small refactoring proposals, and interactive dashboard development assistance.

## 2) Purpose and scope of use
AI tools were used to:
- **Code quality**: Debugging assistance, code review suggestions, naming conventions, refactoring patterns.
- **Learning**: Understanding Optuna, LaTeX formatting, scikit-learn conventions.
- **Documentation**: Writing docstrings, adding comments, improving code clarity.
- **Writing assistance**: Grammar corrections, sentence restructuring for clarity (without changing technical meaning), LaTeX formatting.
- **Dashboard development**: Technical implementation support for creating an interactive Streamlit visualization interface.

**Important distinction**: AI assisted with **linguistic form** (grammar, clarity, formatting) and **technical implementation** (code structure, styling) but **not with intellectual content** (research design, analysis, interpretation, data selection). All substantive technical content and scientific reasoning is the author's original work.

AI tools were NOT used to:
- Generate research ideas, hypotheses, or experimental design.
- Fabricate results or invent interpretations.
- Write substantive technical content beyond language refinement.
- Decide what data, metrics, or visualizations to present.

## 3) What AI contributed (concrete examples)

### Report writing
- **Before AI**: "The models were trained and we got good results."
- **After AI refinement**: "Supervised models demonstrate superior fraud detection when trained on labeled data with class weighting."
- Reformulated abstract for conciseness (reduced from ~250 to ~200 words).
- Suggested consistent terminology (e.g., "anomaly detection methods" instead of mixing "unsupervised models").

### Methodology improvements
- Identified potential data leakage risk when threshold optimization was initially planned on test set.
- Recommended strict validation/test separation with validation-only calibration.

### Code quality
- Recommended adding docstrings to `get_optimal_threshold_f1()` function.
- Suggested consistent naming convention (e.g., `_semi` suffix for semi-supervised models).

### LaTeX formatting
- Helped format complex tables with proper column alignment.
- Suggested using `\texttt{}` for code references and `\cite{}` for bibliography.
- Recommended figure placement with `[H]` for better control.

### Interactive Dashboard Development (Streamlit)

**AI assistance** (Claude, Anthropic):
- Provided technical implementation support for creating an interactive Streamlit dashboard (`dashboard/final_Dashboard.py`).
- Assisted with Plotly visualization code (bar charts, scatter plots, pie charts), CSS styling, and Streamlit layout structure.
- Helped debug display issues such as chart margins, text positioning within bars, and sidebar styling.

**Author's original decisions and intellectual contributions**:
- **Content selection**: All dashboard pages, sections, metrics, and information presented were entirely defined by the author.
- **Data presentation**: Complete choice of which metrics to display (F1-score, Precision, Recall, ROC-AUC), which visualizations to include, and how to compare model performance.
- **Design choices**: Overall structure (7 pages: Home, Executive Summary, Dataset Overview, Methodology, Models Evaluated, Performance Results, Technical Details), information hierarchy, and user experience flow were entirely author-directed.
- **Technical decisions**: Selection of 54 features to display, identification of 6 key features to highlight, decision to use expandable sections, and how to present model comparisons.
- **Scientific content**: All performance data, model descriptions, research questions, key findings, limitations, and technical interpretations are the author's original analytical work.
- **Analytical insights**: All conclusions drawn from visualizations (e.g., "why ROC-AUC is misleading", "supervised models outperform", ensemble trade-offs) are author's original insights.

**Key distinction**: AI provided the *technical scaffolding* (Streamlit code implementation, Plotly syntax, CSS styling, layout components) while the author determined all *intellectual content* (what to show, which data to visualize, how to analyze, what conclusions to draw, which findings to emphasize). The dashboard serves solely as an interactive visualization tool for results generated entirely by the author's original code, experiments, and analytical interpretations.

## 4) Critical evaluation and AI limitations
AI tools occasionally:
- Suggested overly verbose formulations that were simplified by the author.
- Proposed methodological approaches that required adaptation to the project context.
- Made minor factual errors that were corrected through manual verification.

All AI suggestions were critically evaluated and adapted before inclusion.

## 5) Verification and validation steps
To ensure correctness:
- All numerical values reported (dataset counts, metrics, AUC) were taken from program outputs generated by the final code version.
- Figures included in the report were generated by the project code; AI tools did not generate results.
- Dashboard displays only results produced by author's original machine learning pipeline.

## 6) Author's original contributions
The following elements are entirely the author's work:
- **Research design**: Definition of research questions, choice of models, and experimental protocol.
- **Implementation**: All Python code (data processing, model training, evaluation pipeline, visualization).
- **Experimentation**: Running all experiments, hyperparameter tuning, and generating all results.
- **Analysis**: Interpretation of results, identification of patterns, and drawing conclusions.
- **Figures and tables**: All figures and performance tables generated from author's code outputs.
- **Dashboard content**: All data, metrics, comparisons, and insights presented in the interactive dashboard.

AI tools assisted primarily with **linguistic refinement**, **methodological validation**, and **technical implementation** (dashboard coding), not with intellectual content creation or data analysis.

## 7) Dates / notes (optional)
- Main usage period: December 2025.
- Notes: Used mainly for language polishing, methodology review, and dashboard technical implementation.

## 8) Final declaration
The author acknowledges AI assistance as a **writing, review, and technical implementation tool** while affirming full responsibility for:
- The intellectual content and scientific contributions of this work
- The correctness of all reported results and interpretations
- The originality and integrity of the research methodology
- All code implementation and experimental outcomes
- All analytical decisions regarding what data and insights to present

This disclosure is provided in the spirit of academic transparency and follows emerging best practices for AI tool usage in research contexts.

## Author and Contact

**Dylan Fernandez**  
**Student ID**: [20428967]  
University of Lausanne  
Dylan.Fernandez@unil.ch
