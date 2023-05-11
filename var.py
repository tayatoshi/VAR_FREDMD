import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('seaborn')


# tcode　preprocess
def tcode_preprocess(df: pd.DataFrame, tcode: int) -> tuple[np.ndarray, pd.Index]:
    date = df.index
    data = np.array(df)
    if tcode == 1:
        return data, date
    elif tcode == 2:
        results = np.diff(data, axis=0)
        return results, date[1:]
    elif tcode == 3:
        results = np.diff(np.diff(np.log(data), axis=0), axis=0)
        return results, date[2:]
    elif tcode == 4:
        results = np.log(data)
        return results, date
    elif tcode == 5:
        results = np.diff(np.log(data), axis=0)
        return results, date[1:]
    elif tcode == 6:
        results = np.diff(np.diff(np.log(data), axis=0), axis=0)
        return results, date[2:]
    elif tcode == 7:
        results = np.diff((data[1:]/data[0:-1])-1, axis=0)
        return results, date[2:]
    else:
        raise ValueError('tcodeは1から7の自然数です')


class VARmodel(object):
    def __init__(self,
                 data: np.ndarray
                 ) -> None:
        self.data = data
        self.data_nrow = data.shape[0]
        self.data_ncol = data.shape[1]

    def IC_order(self, maxLag: int, h: int = 0) -> dict[str, pd.DataFrame | int | int]:
        # k variables VAR(p)
        # y_{t+h}=y_{t-1}+...+y_{t-p}
        # maxLag>=1
        # T >= k*k*p + k
        T = self.data_nrow - maxLag
        ic_results = pd.DataFrame(np.zeros([maxLag, 2]), columns=['AIC', 'BIC'], index=range(1, maxLag+1))
        y_mat = self.data[maxLag+h:]
        y_vec = y_mat.reshape(y_mat.shape[0]*y_mat.shape[1], 1, order='F')
        for p in range(1, maxLag+1):
            x = np.c_[np.ones(self.data_nrow-h-maxLag), self.data[maxLag-1:self.data_nrow-1-h]]
            if maxLag > 1:
                for lag in range(2, p+1):
                    x = np.c_[x, self.data[maxLag-lag:self.data_nrow-lag-h]]
            eye_ncol = np.eye(self.data_ncol)
            xx = np.kron(eye_ncol, x)
            phi_vec = np.linalg.inv(xx.T @ xx) @ xx.T @ y_vec
            # c1, y1t-1, y2t-1, y3t-1, y1t-2, y2t-2, y3t-2
            # c2, y1t-1, y2t-1, y3t-1, y1t-2, y2t-2, y3t-2
            # c3, y1t-1, y2t-1, y3t-1, y1t-2, y2t-2, y3t-2
            residuals = (y_vec - xx @ phi_vec).reshape(self.data_ncol-maxLag-h, self.data_ncol, order='F')
            Sigma_hat = residuals.T @ residuals/T
            aic = np.log(np.linalg.det(Sigma_hat)) + 2*(self.data_ncol*self.data_ncol*p + self.data_ncol) / T
            bic = np.log(np.linalg.det(Sigma_hat)) + (self.data_ncol*self.data_ncol*p + self.data_ncol) * (np.log(T)/T)
            ic_results.loc[p, 'AIC'] = aic
            ic_results.loc[p, 'BIC'] = bic
        lag_aic = int(ic_results['AIC'].idxmin())
        lag_bic = int(ic_results['BIC'].idxmin())
        ic_results.loc[lag_aic, 'AIC'] = str(ic_results.loc[lag_aic, 'AIC']) + '*'
        ic_results.loc[lag_bic, 'BIC'] = str(ic_results.loc[lag_bic, 'BIC']) + '*'
        return {'ic_results': ic_results, 'aic': lag_aic, 'bic': lag_aic}

    def fit(self,
            p: int,
            h: int,
            ic: str = 'aic'
            ) -> pd.DataFrame:
        y_mat = self.data[p+h:]
        y_vec = y_mat.reshape(y_mat.shape[0]*y_mat.shape[1], 1, order='F')
        x = np.c_[np.ones(self.data_nrow-p-h), self.data[p-1:self.data_nrow-1-h]]
        if p > 1:
            for lag in range(2, p+1):
                x = np.c_[x, self.data[p-lag:self.data_nrow-lag-h]]
        eye_ncol = np.eye(self.data_ncol)
        xx = np.kron(eye_ncol, x)
        phi_vec = np.linalg.inv(xx.T @ xx) @ xx.T @ y_vec
        # c1, y1t-1, y2t-1, y3t-1, y1t-2, y2t-2, y3t-2
        # c2, y1t-1, y2t-1, y3t-1, y1t-2, y2t-2, y3t-2
        # c3, y1t-1, y2t-1, y3t-1, y1t-2, y2t-2, y3t-2
        phi_mat = phi_vec.reshape(self.data_ncol, 1+(self.data_ncol*p))
        phi_mat_df = pd.DataFrame(phi_mat)
        col = ['b'+str(i) for i in phi_mat_df.columns]
        index = ['y'+str(i+1) for i in phi_mat_df.index]
        phi_mat_df.columns = col
        phi_mat_df.index = index
        return phi_mat_df

    def forecast(self, p: int, h: int, forecast_length: int) -> np.ndarray:
        y = np.r_[self.data[-p-h:], np.zeros([forecast_length, self.data_ncol])]
        phi = np.array(self.fit(p, h))
        for t in range(forecast_length):
            x = np.flipud(y[t:t+p]).reshape([self.data_ncol*p, 1])
            c = phi[:, 0].reshape([self.data_ncol, 1])
            y[t+p+h] = (c + phi[:, 1:] @ x).T
        forecasts = y[p+h:]
        return forecasts
