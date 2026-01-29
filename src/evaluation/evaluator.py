from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, min_site_num: int = 2, min_date_num: int = 2):
        self.min_site_num = min_site_num
        self.min_date_num = min_date_num

    def evaluate_by_date(self, pred: np.ndarray, true: np.ndarray,
                         masks: np.ndarray, dates: List[str]) -> pd.DataFrame:
        pred = pred.flatten()
        true = true.flatten()
        masks = masks.flatten()

        data = {
            'Date': dates,
            'Pred': pred,
            'True': true,
            'Mask': masks,
        }

        df = pd.DataFrame(data)
        df_result = pd.DataFrame(
            df.groupby('Date').apply(self._calc_metrics_by_date, include_groups=False).dropna().tolist())
        df_result = df_result.sort_values('Corr_R2', ascending=False, na_position='last')
        return df_result

    def _calc_metrics_by_date(self, group):
        pred = group['Pred'].values
        true = group['True'].values
        mask = group['Mask'].values

        if mask.sum() < self.min_site_num:
            return None

        result = {
            'Date': group.name,
            'Total_Points': len(group),
            'Valid_Points': int(mask.sum())
        }

        orig_metrics = self.calc_metrics(pred, true, mask)
        result.update({f'Orig_{k}': v for k, v in orig_metrics.items()})

        true_valid = true[mask > 0]
        systematic_bias = np.mean(pred[mask > 0] - true_valid)
        pred_corrected = pred - systematic_bias
        corr_metrics = self.calc_metrics(pred_corrected, true, mask)
        result.update({f'Corr_{k}': v for k, v in corr_metrics.items()})
        return result

    def evaluate_by_site(self, pred: np.ndarray, true: np.ndarray,
                         masks: np.ndarray, dates: List[str],
                         rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
        pred = pred.flatten()
        true = true.flatten()
        masks = masks.flatten()
        rows = rows.flatten()
        cols = cols.flatten()

        data = {
            'Row': rows,
            'Col': cols,
            'Date': dates,
            'Pred': pred,
            'True': true,
            'Mask': masks,
        }

        df = pd.DataFrame(data)
        df_valid = df[df['Mask'] > 0].copy()
        results = df_valid.groupby(['Row', 'Col']).apply(self._calc_metrics_by_site,
                                                         include_groups=False).dropna()
        df_result = pd.DataFrame(list(results))
        df_result = df_result.sort_values('Corr_R2', ascending=False, na_position='last')
        return df_result

    def _calc_metrics_by_site(self, group):
        pred = group['Pred'].values
        true = group['True'].values
        mask = group['Mask'].values

        if len(group) < self.min_date_num:
            return None

        row, col = group.name
        result = {
            'Row': int(row),
            'Col': int(col),
            'Valid_Dates': len(group)
        }

        orig_metrics = self.calc_metrics(pred, true, mask)
        result.update({f'Orig_{k}': v for k, v in orig_metrics.items()})

        true_valid = true[mask > 0]
        systematic_bias = np.mean(pred[mask > 0] - true_valid)
        pred_corrected = pred - systematic_bias
        corr_metrics = self.calc_metrics(pred_corrected, true, mask)
        result.update({f'Corr_{k}': v for k, v in corr_metrics.items()})
        return result

    @staticmethod
    def _valid(pred, true, mask=None):
        if mask is not None:
            pred, true = pred[mask > 0], true[mask > 0]
        return (pred, true) if len(pred) >= 2 else (None, None)

    @staticmethod
    def bias(pred, true, mask=None):
        p, t = Evaluator._valid(pred, true, mask)
        return np.mean(p - t) if p is not None else np.nan

    @staticmethod
    def ubrmse(pred, true, mask=None):
        p, t = Evaluator._valid(pred, true, mask)
        if p is None:
            return np.nan
        mse = np.mean((p - t) ** 2)
        b = np.mean(p - t)
        return np.sqrt(np.clip(mse - b ** 2, 0, None))

    @staticmethod
    def r2(pred, true, mask=None):
        p, t = Evaluator._valid(pred, true, mask)
        if p is None:
            return np.nan
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        if ss_tot <= 1e-8:
            return 1.0 if ss_res < 1e-8 else np.nan
        r2_val = 1 - ss_res / ss_tot
        return r2_val if np.isfinite(r2_val) else np.nan

    @staticmethod
    def r(pred, true, mask=None):
        p, t = Evaluator._valid(pred, true, mask)
        if p is None or np.std(p) < 1e-12 or np.std(t) < 1e-12:
            return np.nan
        return np.corrcoef(p, t)[0, 1]

    @staticmethod
    def slope(pred, true, mask=None):
        p, t = Evaluator._valid(pred, true, mask)
        if p is None:
            return np.nan
        v = np.var(t)
        if v <= 1e-8:
            return np.nan
        s = np.cov(p, t)[0, 1] / v
        return s if np.isfinite(s) else np.nan

    @staticmethod
    def calc_metrics(pred, true, mask=None):
        return {
            'ubRMSE': Evaluator.ubrmse(pred, true, mask),
            'Bias': Evaluator.bias(pred, true, mask),
            'R2': Evaluator.r2(pred, true, mask),
            'Slope': Evaluator.slope(pred, true, mask),
        }

    def evaluate_by_spatial_distribution(self, df_site_results: pd.DataFrame, height: int, width: int,
                                         figsize: tuple = (16, 6)):
        rows = df_site_results['Row'].values
        cols = df_site_results['Col'].values
        error_values = df_site_results['Corr_ubRMSE'].values
        r2_values = df_site_results['Corr_R2'].values

        n_points = len(rows)
        use_scatter = n_points < height * width * 0.01

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if use_scatter:
            scatter1 = ax1.scatter(cols, rows, c=error_values, cmap='YlOrRd', s=50, edgecolors='black', linewidths=0.5)
            ax1.set_title('ubRMSE (Pred vs InSitu)')
            ax1.set_xlabel('Column Index')
            ax1.set_ylabel('Row Index')
            ax1.set_aspect('equal', adjustable='box')
            plt.colorbar(scatter1, ax=ax1, label='ubRMSE')

            scatter2 = ax2.scatter(cols, rows, c=r2_values, cmap='cividis', s=50, edgecolors='black', linewidths=0.5,
                                   vmin=0, vmax=1)
            ax2.set_title('R² Score (Pred vs InSitu)')
            ax2.set_xlabel('Column Index')
            ax2.set_ylabel('Row Index')
            ax2.set_aspect('equal', adjustable='box')
            plt.colorbar(scatter2, ax=ax2, label='R²')
        else:
            error_grid = self._build_metric_grid(error_values, rows, cols, height, width)
            im1 = ax1.imshow(error_grid, cmap='YlOrRd', aspect='auto', origin='upper')
            ax1.set_title('ubRMSE (Pred vs InSitu)')
            ax1.set_xlabel('Column Index')
            ax1.set_ylabel('Row Index')
            plt.colorbar(im1, ax=ax1, label='ubRMSE')

            r2_grid = self._build_metric_grid(r2_values, rows, cols, height, width)
            im2 = ax2.imshow(r2_grid, cmap='cividis', aspect='auto', origin='upper', vmin=0, vmax=1)
            ax2.set_title('R² Score (Pred vs InSitu)')
            ax2.set_xlabel('Column Index')
            ax2.set_ylabel('Row Index')
            plt.colorbar(im2, ax=ax2, label='R²')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _build_metric_grid(metric_values: np.ndarray, rows: np.ndarray, cols: np.ndarray, height: int,
                           width: int) -> np.ndarray:
        rows = rows.astype(np.int32)
        cols = cols.astype(np.int32)
        metric_grid = np.full((height, width), np.nan, dtype=np.float32)
        for i in range(len(rows)):
            metric_grid[rows[i], cols[i]] = metric_values[i]
        return metric_grid

    def evaluate_overall(self, pred: np.ndarray, true: np.ndarray,
                         masks: np.ndarray) -> dict:
        pred = pred.flatten()
        true = true.flatten()
        masks = masks.flatten()

        valid = masks > 0
        pred_valid = pred[valid]
        true_valid = true[valid]

        if len(pred_valid) == 0:
            return {'Error': 'No valid data points'}

        metrics = self.calc_metrics(pred_valid, true_valid)
        n_valid = len(pred_valid)
        mse = np.mean((pred_valid - true_valid) ** 2)
        rmse = np.sqrt(mse)

        return {
            'Total_Valid_Points': n_valid,
            'RMSE': rmse,
            **metrics
        }

    def print_overall(self, pred: np.ndarray, true: np.ndarray,
                      masks: np.ndarray, title: str = "Overall Evaluation"):
        metrics = self.evaluate_overall(pred, true, masks)

        if 'Error' in metrics:
            print(f"\n{title}: {metrics['Error']}")
            return

        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")
        print(f"Total Valid Points: {metrics['Total_Valid_Points']:,}")
        print(f"\nMetrics:")
        print(f"  RMSE:      {metrics['RMSE']:.6f}")
        print(f"  ubRMSE:    {metrics['ubRMSE']:.6f}")
        print(f"  Bias:      {metrics['Bias']:.6f}")
        print(f"  R²:        {metrics['R2']:.4f}")
        print(f"  Slope:     {metrics['Slope']:.4f}")
        print(f"{'=' * 60}")
