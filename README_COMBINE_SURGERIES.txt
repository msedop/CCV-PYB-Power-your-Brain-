CROSS-SURGERY ANALYSIS: EEG RATIOS + SPECTRAL POWER + wPLI2 + PHYSIOLOGY
==============================================================================

1. Process each surgery using the existing four-file CCV EEG pipeline.
   It must generate, inside each surgery session directory:

   Outputs/Results/EEG_Ratios/EEG_Event_TAR_Engagement_ROI_Power.csv
   Outputs/Results/Coherence_Results_wPLI2/wpli2_stage_ROI_summary.csv
           (or Outputs/Results/wPLI2_Stage_Analysis/wpli2_stage_ROI_summary.csv)
   Outputs/Results/merged_df.xlsx

2. Place all surgery folders somewhere under one shared parent, for example:

   C:\Users\msedo\Documents\Power Your Brain\
       Surgery_001\Outputs\Results\...
       Surgery_002\Outputs\Results\...
       Surgery_003\Outputs\Results\...

   This also works with nested surgery directories.

3. Edit ONLY the STUDY_ROOT in combine_surgeries_analysis.py.
   Set STAGE_ALIASES if the same stage has different names across sessions.
   Ensure labels refer to comparable surgical stages BEFORE pooling.

4. Run:
   python combine_surgeries_analysis.py

   Install required packages if missing: numpy pandas scipy matplotlib openpyxl
   For optional surgery/stage-adjusted regression, install statsmodels.

5. Results are written under STUDY_ROOT/Across_Surgeries_Analysis/:
   all_surgeries_event_level.csv
   surgery_quality_report.csv
   by_stage_descriptive_summary.csv
   paired_surgery_stage_comparisons.csv
   eeg_physiology_adjusted_models.csv
   surgery_centered_descriptive_spearman.csv
   stages_*.png
   METHODS_AND_LIMITATIONS.txt

KEY INTERPRETATION AND LIMITATIONS
- Stage is a CATEGORICAL exposure. No Pearson correlation with arbitrary stage IDs.
- Stage averages / plots show individual surgeries and group means.
- Paired comparisons only use surgeries with both stages, with BH adjustment.
- A physiological change is referenced to the first INCLUDED stage per surgery,
  NOT the eyes-closed/eyes-open preoperative baseline. For true physiological
  baseline normalization, preserve the baseline stages in your input files.
- Repeated stages within the same surgery are not treated as independent people.
- Surgery+stage-adjusted physiological associations require enough surgeries and
  model identifiability; results will remain exploratory.
- Signed debiased squared wPLI estimates are retained without abs() conversion.
- No raw EEG or wPLI is recomputed.
- If sessions have repeated stage names and merged_df.xlsx lacks an event_index,
  add a unique event index to the saved table; matching otherwise is unsafe.
- The PCA-fixed main EEG pipeline retains its own earlier limitations; this
  companion analysis does not amend your upstream preprocessing.
- No clinical or causal inference should rely on the script alone.
