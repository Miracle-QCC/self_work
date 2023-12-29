import matplotlib.pyplot as plt
import pandas as pd

if __name__=="__main__":
    df_origin = pd.read_csv('./1_rounds_original_result.csv')
    df_opt = pd.read_csv('./1_rounds_result.csv')
    df_sa = pd.read_csv('./SA_best.csv')
    df_ga = pd.read_csv('./GA_best.csv')
    df_pos = pd.read_csv('./POS_best.csv')

    plt.figure()

    plt.title(f'Compare')
    plt.plot(df_origin['best'].tolist(),label='ACO')
    plt.plot(df_opt['best'].tolist(),label='OPT-ACO')
    plt.plot(df_sa['best'].to_list(),label='SA')
    plt.plot(df_ga['best'].to_list(), label='GA')
    plt.plot(df_pos['best'].to_list(), label='POS')

    plt.grid(True)
    plt.legend()
    plt.show()