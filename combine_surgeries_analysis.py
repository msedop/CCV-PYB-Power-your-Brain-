"""Cross-surgery EEG + debiased-squared-wPLI + physiology analysis.

Run AFTER each surgery has been processed with the CCV pipeline.
Adjust only STUDY_ROOT and (optionally) STAGE_ALIASES in CONFIG.

Inputs for each session, under SESSION/Outputs/Results/:
  EEG_Ratios/EEG_Event_TAR_Engagement_ROI_Power.csv
  Coherence_Results_wPLI2/wpli2_stage_ROI_summary.csv  OR
  wPLI2_Stage_Analysis/wpli2_stage_ROI_summary.csv
  merged_df.xlsx   (contains per-event physiological means and wPLI/power).

This module DOES NOT estimate connectivity or recalculate PSD. Data from distinct
surgeries are never concatenated as raw EEG. All inference is exploratory.
Requires: numpy, pandas, scipy, matplotlib, openpyxl (for reading xlsx),
optional statsmodels (for surgery/stage-adjusted regression).
"""
from pathlib import Path
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

# ======================== CONFIG: EDIT HERE ================================
STUDY_ROOT = Path(r"C:\Users\msedo\Documents\Power Your Brain")
OUTPUT_DIR = STUDY_ROOT / 'Across_Surgeries_Analysis'

# Optional: harmonize equivalent event names from different sessions.
# Each key is a name as it appears in your Excel surgical annotations.
# E.g. 'Inicio circulación extracorpórea': 'Inicio CEC'
STAGE_ALIASES = {
    # 'Inicio circulación extracorpórea': 'Inicio CEC',
    # 'Clampaje de aorta': 'Clampaje Aorta',
}

# Most useful prespecified variables, rather than fishing across every column.
EEG_METRICS = [
    'TAR_FzTheta_over_PzAlpha',
    'Engagement_Global_Beta_over_ThetaPlusAlpha',
    'Engagement_ROI_PosteriorBeta_over_FrontalThetaPlusParietalAlpha',
    'Relative_Theta_Global_pct',
    'Relative_Alpha_Global_pct',
    'Relative_Theta_Frontal_ROI_pct',
    'Relative_Alpha_Parietal_ROI_pct',
    'Theta_Frontal_ROI_uV2',
    'Alpha_Parietal_ROI_uV2',
    'Alpha_Posterior_ROI_uV2',
    'Beta_Posterior_ROI_uV2',
]
PHYS_PARAMS = ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PR(bpm)']

# Minimum number of paired surgeries for paired tests and number for regressions.
MIN_PAIRED_SURGERIES = 3
MIN_REGRESSION_SURGERIES = 4
# ===========================================================================


def norm(s):
    s = unicodedata.normalize('NFKD', str(s).strip().casefold())
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r'\s+', ' ', s)


ALIASES = {norm(k): str(v).strip() for k, v in STAGE_ALIASES.items()}


def stage_name(s):
    return ALIASES.get(norm(s), str(s).strip())


def find_sessions(root):
    """Find output directories by exact ratio-table presence (no duplicate discovery)."""
    matches = sorted(root.rglob('EEG_Ratios/EEG_Event_TAR_Engagement_ROI_Power.csv'))
    sessions = []
    for file in matches:
        results = file.parent.parent
        if results.name != 'Results' or results.parent.name != 'Outputs':
            continue
        session = results.parent.parent
        sessions.append((session, results))
    return sessions


def require_single_stage(df, table_name):
    if df['_event_index'].isna().any():
        raise ValueError(f'{table_name} contains missing/invalid event indices.')
    if df['_event_index'].duplicated().any():
        raise ValueError(f'{table_name} has duplicate event indices. Check event extraction.')


def prepare_ratios(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if not {'Event_index', 'Event'}.issubset(df.columns):
        raise ValueError(f'{path}: missing Event_index/Event')
    df['_event_index'] = pd.to_numeric(df['Event_index'], errors='coerce')
    require_single_stage(df, str(path))
    df = df.rename(columns={'Event': 'EEG_Event_Label'})
    return df.drop(columns=['Event_index'])


def prepare_wpli(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    needed = {'event_index', 'event_label', 'band'}
    if not needed.issubset(df.columns):
        raise ValueError(f'{path}: required {needed}')
    df['band'] = df['band'].astype(str).str.casefold()
    selected = df.loc[df['band'] == 'theta'].copy()
    selected['_event_index'] = pd.to_numeric(selected['event_index'], errors='coerce')
    require_single_stage(selected, str(path))
    wanted = [
        'n_epochs', 'n_channels', 'Primary_FP_Theta_Mean_dwPLI2',
        'Primary_FP_Theta_N_Pairs', 'Frontoparietal_Mean_dwPLI2',
        'Within_Frontal_Mean_dwPLI2', 'Fz_Pz_Mean_dwPLI2'
    ]
    keep = ['_event_index', 'event_label'] + [c for c in wanted if c in selected.columns]
    selected = selected[keep].rename(columns={'event_label': 'WPLI_Event_Label'})
    return selected


def prepare_physiology(path):
    df = pd.read_excel(path)
    label_col = 'event_label' if 'event_label' in df.columns else 'Event' if 'Event' in df.columns else None
    if label_col is None:
        raise ValueError(f'{path}: no event_label or Event column')
    mean_cols = [f'{p}_mean' for p in PHYS_PARAMS if f'{p}_mean' in df.columns]
    if not mean_cols:
        raise ValueError(f'{path}: no physiological *_mean columns')
    keep = [label_col] + mean_cols
    if 'event_index' in df.columns:
        keep.insert(1, 'event_index')
    df = df[keep].copy()
    df = df.rename(columns={label_col: 'Phys_Event_Label'})
    if 'event_index' in df:
        df['_event_index'] = pd.to_numeric(df.pop('event_index'), errors='coerce')
    return df


def reconcile_physiology(left, phys, filename):
    """Join physiological means by verified event index, or unique normalized name."""
    if '_event_index' in phys:
        require_single_stage(phys, filename)
        right = phys.copy()
        combo = left.merge(right, on='_event_index', how='left', validate='one_to_one')
        for a, b in zip(combo['EEG_Event_Label'], combo['Phys_Event_Label']):
            if pd.notna(b) and norm(a) != norm(b):
                warnings.warn(f'Physiology event mismatch for {filename}: {a!r} vs {b!r}')
        return combo
    # Only labels remain: reject duplicate labels, so events cannot be silently mismatched.
    left = left.copy()
    phys = phys.copy()
    left['_stagekey'] = left['EEG_Event_Label'].map(norm)
    phys['_stagekey'] = phys['Phys_Event_Label'].map(norm)
    if left['_stagekey'].duplicated().any() or phys['_stagekey'].duplicated().any():
        raise ValueError(
            f'{filename}: repeated event names, but physiology output has no event_index. '
            'Add event_index to merged_df.xlsx in the main pipeline before combining surgeries.'
        )
    return left.merge(phys, on='_stagekey', how='left', validate='one_to_one').drop(columns='_stagekey')


def load_one_surgery(session, results):
    ratio = results/'EEG_Ratios'/'EEG_Event_TAR_Engagement_ROI_Power.csv'
    wpli_candidates = [
        results/'Coherence_Results_wPLI2'/'wpli2_stage_ROI_summary.csv',
        results/'wPLI2_Stage_Analysis'/'wpli2_stage_ROI_summary.csv',
    ]
    wpli = next((p for p in wpli_candidates if p.exists()), None)
    physiological = results/'merged_df.xlsx'
    if wpli is None:
        warnings.warn(f'Skipping {session}: wPLI2 ROI summary not found')
        return None
    if not physiological.exists():
        warnings.warn(f'Skipping {session}: merged_df.xlsx not found')
        return None
    rr = prepare_ratios(ratio)
    ww = prepare_wpli(wpli)
    # Check event-index to label alignment before merging.
    check = rr[['_event_index', 'EEG_Event_Label']].merge(
        ww[['_event_index', 'WPLI_Event_Label']], on='_event_index', how='outer', indicator=True,
        validate='one_to_one')
    if (check['_merge'] != 'both').any():
        raise ValueError(f'{session}: wPLI and ratio event indices disagree; inspect segmentation')
    mismatches = check.loc[
        [norm(a) != norm(b) for a,b in zip(check.EEG_Event_Label,check.WPLI_Event_Label)]
    ]
    if not mismatches.empty:
        raise ValueError(f'{session}: wPLI and ratios have different event labels at same indices.\n'
                         f'{mismatches.to_string(index=False)}')
    merged = rr.merge(ww, on='_event_index', how='inner', validate='one_to_one')
    merged = reconcile_physiology(merged, prepare_physiology(physiological), str(physiological))
    if merged['Phys_Event_Label'].isna().all():
        raise ValueError(f'{session}: no physiology stages matched EEG stages. Check labels.')
    if merged['Phys_Event_Label'].isna().any():
        warnings.warn(f'{session}: some physiological stage means are unmatched; see combined table.')
    merged = merged.sort_values('_event_index').reset_index(drop=True)
    merged['SurgeryID'] = str(session.relative_to(STUDY_ROOT)).replace('\\', '/')
    merged['Stage'] = merged['EEG_Event_Label'].map(stage_name)
    merged['Stage_key'] = merged['Stage'].map(norm)
    merged['Stage_order'] = np.arange(1, len(merged)+1)
    return merged


def add_physiology_changes(df):
    """Reference=first INCLUDED stage in each surgery, not EC/EO baseline."""
    df = df.copy().sort_values(['SurgeryID', 'Stage_order'])
    for param in PHYS_PARAMS:
        src = f'{param}_mean'
        if src not in df:
            continue
        x = pd.to_numeric(df[src], errors='coerce')
        df[src] = x
        by = df.groupby('SurgeryID', sort=False)[src]
        first_row = df.groupby('SurgeryID',sort=False)[src].transform(lambda z: z.iloc[0])
        previous = by.shift(1)
        df[f'{param}_change_vs_first'] = x-first_row
        df[f'{param}_change_vs_previous'] = x-previous
        df[f'{param}_pct_vs_first'] = 100 * (x-first_row)/first_row.where(first_row.abs()>1e-12)
    return df


def bh_fdr(pvalues):
    """Benjamini-Hochberg across available p-values; missing stays NaN."""
    vals = np.asarray(pvalues,float)
    good = np.isfinite(vals)
    out = np.full(len(vals),np.nan)
    m = int(good.sum())
    if not m: return out
    idx = np.flatnonzero(good)
    ordered = idx[np.argsort(vals[idx])]
    q = vals[ordered]*m/np.arange(1,m+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[ordered] = np.minimum(1,q)
    return out


def tidy_metrics(df):
    columns = [c for c in EEG_METRICS if c in df]
    columns += [c for c in ('Primary_FP_Theta_Mean_dwPLI2','Within_Frontal_Mean_dwPLI2',
                           'Frontoparietal_Mean_dwPLI2','Fz_Pz_Mean_dwPLI2') if c in df]
    columns += [f'{p}_mean' for p in PHYS_PARAMS if f'{p}_mean' in df]
    return columns


def stage_summary(df, metrics):
    rows=[]
    stage_order = df.groupby('Stage_key')['Stage_order'].median().sort_values()
    for key in stage_order.index:
        sub=df[df.Stage_key==key]
        for metric in metrics:
            v = pd.to_numeric(sub[metric], errors='coerce').dropna()
            rows.append({
                'Stage':sub.Stage.iloc[0], 'Stage_key':key,
                'Stage_order_median':float(stage_order[key]),
                'Metric':metric, 'N_surgeries':int(sub.loc[v.index,'SurgeryID'].nunique()),
                'Mean':float(v.mean()) if len(v) else np.nan,
                'SD':float(v.std(ddof=1)) if len(v)>1 else np.nan,
                'Median':float(v.median()) if len(v) else np.nan,
                'Min':float(v.min()) if len(v) else np.nan,
                'Max':float(v.max()) if len(v) else np.nan,
            })
    return pd.DataFrame(rows)


def paired_stage_tests(df, metrics):
    """Each surgery contributes at most one observation per stage."""
    rows=[]
    # Earliest commonly present stage; reference is chosen globally, not per surgery.
    stages = df.groupby('Stage_key')['Stage_order'].median().sort_values().index.tolist()
    if not stages: return pd.DataFrame()
    ref = stages[0]
    for metric in metrics:
        tab = df.pivot_table(index='SurgeryID', columns='Stage_key', values=metric, aggfunc='first')
        for stage in stages[1:]:
            if ref not in tab or stage not in tab: continue
            pairs=tab[[ref,stage]].apply(pd.to_numeric,errors='coerce').dropna()
            d=pairs[stage]-pairs[ref]
            p=np.nan
            if len(d)>=MIN_PAIRED_SURGERIES and np.any(np.abs(d)>1e-12):
                try: p=float(wilcoxon(d,zero_method='wilcox').pvalue)
                except ValueError: pass
            rows.append({'Metric':metric,'Reference_stage':ref,'Comparison_stage':stage,
                         'N_paired_surgeries':len(d), 'Mean_delta':d.mean() if len(d) else np.nan,
                         'Median_delta':d.median() if len(d) else np.nan,'Wilcoxon_p_exploratory':p})
    out=pd.DataFrame(rows)
    if len(out): out['BH_q_exploratory']=bh_fdr(out.Wilcoxon_p_exploratory)
    return out


def associations(df, eeg_metrics, physiological_covariates):
    """Cluster-robust surgery+stage-adjusted associations, where identifiable.

    Model: EEG_marker ~ physiological_change + C(SurgeryID) + C(Stage_key)
    Coefficient interpreted as adjusted association, not causality.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        warnings.warn('statsmodels unavailable: skipping adjusted regression; pip install statsmodels')
        return pd.DataFrame()
    rows=[]
    for outcome in eeg_metrics:
        for predictor in physiological_covariates:
            if outcome not in df or predictor not in df:continue
            tmp=df[['SurgeryID','Stage_key', outcome,predictor]].copy().rename(
                columns={outcome:'y',predictor:'x'})
            tmp['y']=pd.to_numeric(tmp['y'],errors='coerce')
            tmp['x']=pd.to_numeric(tmp['x'],errors='coerce')
            tmp=tmp.replace([np.inf,-np.inf],np.nan).dropna()
            n_surg=tmp.SurgeryID.nunique()
            if n_surg<MIN_REGRESSION_SURGERIES or len(tmp)<12 or tmp['x'].nunique()<3:
                continue
            # Ensure predictor varies beyond surgery and stage indicators.
            if not (tmp.groupby('SurgeryID')['x'].nunique()>1).any():continue
            try:
                model=smf.ols('y ~ x + C(SurgeryID) + C(Stage_key)',data=tmp).fit(
                    cov_type='cluster',cov_kwds={'groups':tmp.SurgeryID})
                # Near-collinear designs cannot reliably estimate this association.
                from numpy.linalg import matrix_rank
                if matrix_rank(model.model.exog) < model.model.exog.shape[1]:
                    continue
                rows.append({'EEG_metric':outcome,'Physiology_metric':predictor,
                             'n_observations':len(tmp),'n_surgeries':n_surg,
                             'adjusted_beta':float(model.params['x']),
                             'cluster_robust_SE':float(model.bse['x']),
                             'p_exploratory':float(model.pvalues['x']),
                             'CI95_lower':float(model.conf_int().loc['x',0]),
                             'CI95_upper':float(model.conf_int().loc['x',1])})
            except Exception as exc:
                warnings.warn(f'Adjusted model skipped: {outcome} ~ {predictor}: {exc}')
    out=pd.DataFrame(rows)
    if len(out):out['BH_q_exploratory']=bh_fdr(out['p_exploratory'])
    return out


def cross_surgery_spearman(df, cols):
    """Descriptive correlation of surgery-centered stage values (NOT independent p-values)."""
    centered=df.copy()
    for col in cols:
        centered[col]=pd.to_numeric(centered[col],errors='coerce')
        centered[col]=centered[col]-centered.groupby('SurgeryID')[col].transform('mean')
    rows=[]
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            sub=centered[['SurgeryID',a,b]].replace([np.inf,-np.inf],np.nan).dropna()
            rho=np.nan
            if len(sub)>=4 and sub[a].nunique()>1 and sub[b].nunique()>1:
                rho=float(spearmanr(sub[a],sub[b]).statistic)
            rows.append({'X':a,'Y':b,'rho_descriptive':rho,
                         'n_stage_observations':len(sub),
                         'n_surgeries':sub.SurgeryID.nunique()})
    return pd.DataFrame(rows)


def make_plots(df, summary, output):
    candidates=['Primary_FP_Theta_Mean_dwPLI2','TAR_FzTheta_over_PzAlpha',
                'Relative_Theta_Frontal_ROI_pct','Relative_Alpha_Parietal_ROI_pct',
                'HR(bpm)_mean','SpO2(%)_mean']
    chosen=[col for col in candidates if col in df]
    if not chosen:return
    stages=df.groupby('Stage_key')['Stage_order'].median().sort_values().index.tolist()
    label_lookup=df.drop_duplicates('Stage_key').set_index('Stage_key')['Stage'].to_dict()
    for metric in chosen:
        fig,ax=plt.subplots(figsize=(max(10,len(stages)*1.15),5))
        for surgery, sub in df.groupby('SurgeryID'):
            ordered=sub.set_index('Stage_key').reindex(stages)
            ax.plot(range(len(stages)),ordered[metric].to_numpy(float),marker='o',alpha=.3,lw=1)
        ss=summary[summary.Metric==metric].set_index('Stage_key').reindex(stages)
        ax.plot(range(len(stages)),ss['Mean'].to_numpy(float),marker='D',lw=2.4,label='Across-surgery mean')
        ax.set_xticks(range(len(stages)),[label_lookup[s] for s in stages],rotation=55,ha='right')
        ax.set_ylabel(metric);ax.set_title(f'Surgical-stage trajectories: {metric}')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        safe=re.sub(r'[^a-zA-Z0-9_-]+','_',metric)
        fig.savefig(output/f'stages_{safe}.png',dpi=180,bbox_inches='tight')
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    found=find_sessions(STUDY_ROOT)
    if not found:
        raise FileNotFoundError(f'No processed sessions below {STUDY_ROOT}; '
                                'expect Outputs/Results/EEG_Ratios/EEG_Event_TAR_Engagement_ROI_Power.csv')
    frames=[]; quality=[]
    for session,results in found:
        try:
            frame=load_one_surgery(session,results)
            if frame is None:continue
            frames.append(frame)
            quality.append({'SurgeryID':frame.SurgeryID.iloc[0], 'Stages':len(frame),
                            'Missing_physiology_rows':int(frame[[c for c in frame if c.endswith('_mean') and c.split('_mean')[0] in PHYS_PARAMS]].isna().all(axis=1).sum()),
                            'Missing_primary_wPLI_rows':int(frame.Primary_FP_Theta_Mean_dwPLI2.isna().sum())})
            print('Loaded:',session,'stages:',len(frame))
        except Exception as exc:
            # Strict default: stop rather than silently omit surgeries and bias findings.
            raise RuntimeError(f'Cannot safely combine session {session}: {exc}') from exc
    if not frames:raise RuntimeError('No complete surgery datasets to combine')
    combined=pd.concat(frames,ignore_index=True)
    combined=add_physiology_changes(combined)
    # Repeated label within one surgery cannot be compared as distinct stages by name.
    duplicated=combined.duplicated(['SurgeryID','Stage_key'],keep=False)
    if duplicated.any():
        raise ValueError('Repeated normalized stage labels within surgery: '
                         + str(combined.loc[duplicated,['SurgeryID','Stage','Stage_order']].to_dict('records')))
    metrics=tidy_metrics(combined)
    phys_change=[f'{p}_change_vs_first' for p in PHYS_PARAMS if f'{p}_change_vs_first' in combined]
    combined.to_csv(OUTPUT_DIR/'all_surgeries_event_level.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(quality).to_csv(OUTPUT_DIR/'surgery_quality_report.csv',index=False)
    stage=stage_summary(combined,metrics)
    stage.to_csv(OUTPUT_DIR/'by_stage_descriptive_summary.csv',index=False,encoding='utf-8-sig')
    paired=paired_stage_tests(combined,metrics)
    paired.to_csv(OUTPUT_DIR/'paired_surgery_stage_comparisons.csv',index=False)
    out_metrics=[c for c in EEG_METRICS if c in combined]
    out_metrics += [c for c in ('Primary_FP_Theta_Mean_dwPLI2',) if c in combined]
    models=associations(combined,out_metrics,phys_change)
    models.to_csv(OUTPUT_DIR/'eeg_physiology_adjusted_models.csv',index=False)
    # Exploratory within-surgery EEG ratio vs power, wPLI, and physiology changes.
    descriptive=cross_surgery_spearman(combined,out_metrics+phys_change)
    descriptive.to_csv(OUTPUT_DIR/'surgery_centered_descriptive_spearman.csv',index=False)
    make_plots(combined,stage,OUTPUT_DIR)
    with open(OUTPUT_DIR/'METHODS_AND_LIMITATIONS.txt','w',encoding='utf-8') as f:
        f.write('Each row is one surgery x surgical stage.\n'
                'EEG ratios are Welch-derived and wPLI2 estimates signed debiased squared wPLI.\n'
                'The relative-power denominator is 1-40 Hz per channel.\n'
                'Physiology change vs first uses the first included event per surgery, NOT EC/EO baseline.\n'
                'Stages are CATEGORICAL, not a numeric covariate for Pearson correlation.\n'
                'Paired tests require same surgery contributing both stages; BH exploratory correction.\n'
                'Regression adjusts for surgery and stage with clustered SE, when identifiable.\n'
                'Stage observation counts are NOT independent patient sample sizes.\n'
                'All associations are exploratory; observational and noncausal.\n'
                'Check stage aliases, missing data, epoch counts, EEG cleaning, and timestamps.\n'
                'No single-surgery results were pooled as raw EEG.\n')
    print(f'Completed: {combined.SurgeryID.nunique()} surgeries, {len(combined)} event rows.\n'
          f'Results: {OUTPUT_DIR}')


if __name__=='__main__':
    main()
