from scipy import stats
import copy
    hist_val = avg_value[df_pivot.columns].values
    real_val = df_pivot.values[0]
    observed = pd.DataFrame({"hist": hist_val,
                             "real": real_val}, index=df_pivot.columns).T
    chi_data = copy.deepcopy(observed)
    chi_data["row_sum"] = chi_data.apply(np.sum, axis=1)
    chi_data.loc["col_sum",:] = chi_data.apply(np.sum, axis=0)
    expected = copy.deepcopy(observed)
    for i in range(len(chi_data)-1):
        for j in range(len(chi_data.columns)-1):
            expected.iloc[i,j] = chi_data.iloc[2,j]*(chi_data.iloc[i,3]/chi_data.iloc[2,3])
    chisq_pvalue = stats.chisquare(f_obs=observed, f_exp=expected, ddof=expected.shape[1], axis=None).pvalue # df = k - ddof - 1
