#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired significance checks for Table-V predictions.

The script recomputes selected deployable predictions and reports paired error
reductions, improved-window fractions, Wilcoxon p-values when scipy is available,
and bootstrap confidence intervals.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from fame import FAMEModel
from fame.utils import ensure_datetime, valid_demand_mask, json_dump
from run_table_v import _usff_rules, _selector_prediction, _panel_to_wide
from sklearn.linear_model import Ridge


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='./fame_model')
    ap.add_argument('--history', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out-dir', default='./output/significance')
    ap.add_argument('--device', default=None)
    ap.add_argument('--bootstrap', type=int, default=1000)
    return ap.parse_args()


def _sqerr(actual, pred, cfg):
    pred_col = 'prediction' if 'prediction' in pred.columns else 'predicted_sales'
    m = actual.merge(pred[list(cfg.id_cols)+[cfg.date_col,pred_col]], on=list(cfg.id_cols)+[cfg.date_col], how='inner')
    valid = valid_demand_mask(m, cfg)
    m = m.loc[valid].copy()
    m['sq_error'] = (pd.to_numeric(m[cfg.target_col], errors='coerce').fillna(0) - pd.to_numeric(m[pred_col], errors='coerce').fillna(0)) ** 2
    m['abs_error'] = (pd.to_numeric(m[cfg.target_col], errors='coerce').fillna(0) - pd.to_numeric(m[pred_col], errors='coerce').fillna(0)).abs()
    return m[list(cfg.id_cols)+[cfg.date_col,'sq_error','abs_error']]


def _wilcoxon(x):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(x, alternative='greater').pvalue)
    except Exception:
        # Sign-test fallback using normal approximation for positive improvements.
        pos = np.sum(x > 0); n = np.sum(x != 0)
        if n == 0: return 1.0
        z = (pos - 0.5*n) / max(np.sqrt(0.25*n), 1e-12)
        from math import erf, sqrt
        return float(1 - 0.5*(1 + erf(z/sqrt(2))))


def _compare(name, base_err, fame_err, keys, n_boot=1000, seed=42):
    df = base_err.merge(fame_err, on=keys, suffixes=('_base','_fame'))
    if df.empty:
        return {'comparison': name, 'available': False}
    # Positive means baseline error - FAME error, so greater than zero favors FAME.
    diff = df['sq_error_base'].to_numpy() - df['sq_error_fame'].to_numpy()
    mae_diff = df['abs_error_base'].to_numpy() - df['abs_error_fame'].to_numpy()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diff), len(diff))
        boot.append(float(np.mean(diff[idx])))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return {
        'comparison': name,
        'available': True,
        'n_rows': int(len(df)),
        'mse_reduction_abs': float(np.mean(diff)),
        'mae_reduction_abs': float(np.mean(mae_diff)),
        'improved_fraction': float(np.mean(diff > 0)),
        'wilcoxon_or_sign_p': _wilcoxon(diff),
        'bootstrap_diff_ci95_low': float(lo),
        'bootstrap_diff_ci95_high': float(hi),
    }


def main():
    args=parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model=FAMEModel.load(args.model, device=args.device); cfg=model.config
    hist=ensure_datetime(pd.read_csv(args.history), cfg.date_col); test=ensure_datetime(pd.read_csv(args.test), cfg.date_col)
    actual=test[list(cfg.id_cols)+[cfg.date_col,cfg.target_col]].copy()
    preds_by_expert=model.expert_pool.predict_all(hist, test)
    if 'lightgbm' in preds_by_expert:
        light=preds_by_expert['lightgbm']
    else:
        # use first available expert as fallback for demo mode
        first_name=next(iter(preds_by_expert)); light=preds_by_expert[first_name]
    pred_fame, explain=model.predict(hist, future_df=test, return_explanations=True)
    pred_fame=pred_fame.rename(columns={'predicted_sales':'prediction'})
    # Dense soft MoE baseline.
    panel=[]
    for e,p in preds_by_expert.items():
        q=p.copy(); q['expert']=e; panel.append(q)
    panel=pd.concat(panel, ignore_index=True)
    expert_names=list(model.expert_pool.expert_names)
    hist_prep=model._prepare(hist, complete_grid=True)
    fp=model.fingerprint_extractor.transform(hist_prep, reference_date=hist_prep[cfg.date_col].max())
    probs=model.router.predict_proba(fp)
    dense=panel.merge(probs[list(cfg.id_cols)+expert_names], on=list(cfg.id_cols), how='left')
    dense['w']=dense.apply(lambda r: float(r.get(r['expert'],0)), axis=1)
    denom=dense.groupby(list(cfg.id_cols)+[cfg.date_col])['w'].transform('sum').replace(0,np.nan)
    dense['w']=(dense['w']/denom).fillna(1/max(1,len(expert_names)))
    dense['weighted_prediction']=dense['prediction']*dense['w']
    dense_pred=dense.groupby(list(cfg.id_cols)+[cfg.date_col],as_index=False)['weighted_prediction'].sum().rename(columns={'weighted_prediction':'prediction'})
    keys=list(cfg.id_cols)+[cfg.date_col]
    fame_err=_sqerr(actual,pred_fame,cfg)
    results=[]
    results.append(_compare('FAME_Top-r_vs_LightGBM_or_first_single', _sqerr(actual,light,cfg), fame_err, keys, args.bootstrap))

    # Rule-USFF baseline.
    try:
        fp_raw=model.fingerprint_extractor.raw_transform(hist_prep, reference_date=hist_prep[cfg.date_col].max())
        usff_sel=_usff_rules(fp_raw, expert_names)
        usff_sel.columns=list(cfg.id_cols)+['selected_expert']
        usff_pred=_selector_prediction(panel, usff_sel, cfg, 'rule_based_usff')
        results.append(_compare('FAME_Top-r_vs_RuleUSFF', _sqerr(actual,usff_pred,cfg), fame_err, keys, args.bootstrap))
    except Exception as exc:
        results.append({'comparison':'FAME_Top-r_vs_RuleUSFF','available':False,'error':str(exc)})

    # Stacking baseline trained on oracle-window predictions.
    try:
        if model.oracle_ is not None and model.oracle_.prediction_tensor is not None:
            P=np.asarray(model.oracle_.prediction_tensor,dtype=float)
            Y=np.asarray(model.oracle_.target_tensor,dtype=float)
            Xs=P.transpose(0,2,1).reshape(-1,len(expert_names)); ys=Y.reshape(-1)
            mask=np.isfinite(ys) & np.all(np.isfinite(Xs),axis=1)
            if mask.sum() >= max(5,len(expert_names)):
                stack=Ridge(alpha=1.0).fit(np.nan_to_num(Xs[mask],nan=0.0),ys[mask])
                wide=_panel_to_wide(panel,cfg,expert_names)
                stack_pred=wide[list(cfg.id_cols)+[cfg.date_col]].copy()
                stack_pred['prediction']=stack.predict(wide[expert_names].to_numpy(dtype=float))
                results.append(_compare('FAME_Top-r_vs_Stacking', _sqerr(actual,stack_pred,cfg), fame_err, keys, args.bootstrap))
            else:
                results.append({'comparison':'FAME_Top-r_vs_Stacking','available':False,'error':'not enough oracle rows'})
        else:
            results.append({'comparison':'FAME_Top-r_vs_Stacking','available':False,'error':'missing oracle tensor'})
    except Exception as exc:
        results.append({'comparison':'FAME_Top-r_vs_Stacking','available':False,'error':str(exc)})

    results.append(_compare('FAME_Top-r_vs_DenseSoftMoE', _sqerr(actual,dense_pred,cfg), fame_err, keys, args.bootstrap))
    df=pd.DataFrame(results)
    df.to_csv(out_dir/'significance_tests.csv', index=False)
    json_dump({'results': results}, out_dir/'significance_tests.json')
    print(df.to_string(index=False))


if __name__=='__main__':
    main()
