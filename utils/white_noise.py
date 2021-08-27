import numpy as np

class WhiteNoise():


    @staticmethod
    def isWhiteNoise(series):
        
        '''
        Time series that show no autocorrelation are called white noise

        For a white noise series, we expect 95% of the spikes in the ACF to lie within +- 2 / sqrt(T) 
        where T is the length of the time series.
        '''
        
        auto_corr = np.array([WhiteNoise().acf(series, x) for x in range(len(series))])
        
        # Calculating bound
        upper_bound = 2.0 / np.sqrt(len(auto_corr))
        lower_bound = - 2.0 / np.sqrt(len(auto_corr))
        inBound = len(auto_corr[np.logical_and(auto_corr <= upper_bound, auto_corr >= lower_bound)])
        per_inBound = inBound / len(series)
        
        # if 95% of spikes are within bound, timeseries is white noise
        if per_inBound >= 0.95:
            return True
        else:
            return False


    @staticmethod
    def filter_WN(data, norm_stat):
        # Remove White Noise
        removed = 0
        total = 0
    
        for group in data.keys():
            for FE, res in list(data[group].items()):
                series = [x[1] for x in res]
            
                # Remove time series if time series is white noise
                if WhiteNoise().isWhiteNoise(series):
                    removed += 1
                    del data[group][FE]
                    del norm_stat[group][FE]
                total += 1
        print("removed : " + str(removed) + " total : " + str(total))
        return data, norm_stat, (removed, total)
    
    
    @staticmethod
    def acf(series, lag):
        # Calculate auto correlation between series and lag times series
        
        if lag == 0:
            corr = 1.0
        else:
            mean = np.mean(series)
            var = np.var(series)
            covar = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / len(series)
            corr = covar / var
        return corr
