# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:00:50 2021

@author: pablo
"""

from montecarlo import *
from funciones import cal_yf_from_mat, days_yf, parametros_to_pasos
from math import exp
import numpy as np

def pricer(S0, q0, r, a, b, vols, volq, rho, M, h_previo, h_elegido, mat_position):
    a_pasos = parametros_to_pasos(a, t0, T_futdiv)
    volq_pasos = parametros_to_pasos(volq, t0, T_futdiv)
    vols_pasos = parametros_to_pasos(vols, t0, T_opt_sx5e)

    # Número de pasos N en la simulación
    N = len(h_elegido)
    N_previo = len(h_previo)

    # Cálculos usando normrnd y correlacionando variables después
    S, q = HybridStockDividendsMSamples(S0, q0, r, a_pasos, b, vols_pasos, volq_pasos, rho, M, N, h_elegido, mat_position)

    payoffs_divfut = PayoffDivFut(S[N_previo:N + 1], q[N_previo:N + 1], h_elegido[N_previo:N])
    payoffs_divopt = exp(-r * yf_futdiv[0]) * PayoffOptCall(payoffs_divfut, K_div)
    payoffs_eqopt = exp(-r * yf_futdiv[0]) * PayoffOptCall(S[N], S0)

    return [np.mean(payoffs_divfut), np.mean(payoffs_divopt), np.mean(payoffs_eqopt)]

T_opt_sx5e = ['2020-12-18',
              '2021-12-17',
              '2022-12-16',
              '2023-12-15']
T_futdiv = ['2020-12-18',
            '2021-12-17',
            '2022-12-16',
            '2023-12-15']
T_endyear = ['2020-01-01',
             '2021-01-01',
             '2022-01-01',
             '2023-01-01',
             '2024-01-01']

t0 = '2020-04-01'
days_futdiv = [np.busday_count(t0, fecha) for fecha in T_futdiv]
days_opt_sx5e = [np.busday_count(t0, fecha) for fecha in T_opt_sx5e]

# Year fractions maturities
yf_futdiv = cal_yf_from_mat(t0, T_futdiv, T_endyear)

# Steps for each maturity.
days_yf, h = days_yf(t0, T_futdiv, T_endyear)
mat_position = [len(hs) for hs in h]

h1 = h[0]
h2 = h[1]
h3 = h[2]
h4 = h[3]

div_fut_prices = [53.1, 49.8, 63.6, 67.4]
div_call_opt_prices = [5.60, 4.19, 11.02, 16.13]  # Excel. Tomo el strike 65, que es el más líquido.
eurostoxx_call_opt_prices = [239.0016, 292.9, 312.5604, 336.7692]

objetivos = [[div_fut_prices[i], div_call_opt_prices[i], eurostoxx_call_opt_prices[i]] for i in range(4)]


# Datos del problema
S0 = 2680.3  # También es el strike de las call sobre el eurostoxx
q0 = 0.019967966
M = 2 ** 12
K_div = 65  # Strike dividend call options
rho = -0.189292925
r = -0.00168

# Parametros
b = 0.001  # Criterio experto
a = [0.024632889670984207, 0.03943332940236466, 0.051152059755964985, 0.0559917559120302]
vols = [0.2941956753706126, 0.2115385477314622, 0.2084001707439573, 0.21062649631184904]
volq = [0.17712546432086035, 0.1297972709969229, 0.1263240067709712, 0.1761761314040222]

prices_1 = pricer(S0, q0, r, a, b, vols, volq, rho, M, [1], h1, mat_position)
prices_2 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h1, h2, mat_position)
prices_3 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h2, h3, mat_position)
prices_4 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h3, h4, mat_position)

prices_1_eqbump = pricer(S0*1.01, q0, r, a, b, vols, volq, rho, M, [1], h1, mat_position)
prices_2_eqbump = pricer(S0*1.01, q0, r, a, b, vols, volq, rho, M, h1, h2, mat_position)
prices_3_eqbump = pricer(S0*1.01, q0, r, a, b, vols, volq, rho, M, h2, h3, mat_position)
prices_4_eqbump = pricer(S0*1.01, q0, r, a, b, vols, volq, rho, M, h3, h4, mat_position)

prices_1_divbump = pricer(S0, q0+0.0001, r, a, b, vols, volq, rho, M, [1], h1, mat_position)
prices_2_divbump = pricer(S0, q0+0.0001, r, a, b, vols, volq, rho, M, h1, h2, mat_position)
prices_3_divbump = pricer(S0, q0+0.0001, r, a, b, vols, volq, rho, M, h2, h3, mat_position)
prices_4_divbump = pricer(S0, q0+0.0001, r, a, b, vols, volq, rho, M, h3, h4, mat_position)

eq_delta_1 = [(shift-base)/0.01 for shift, base in zip(prices_1_eqbump, prices_1)]
eq_delta_2 = [(shift-base)/0.01 for shift, base in zip(prices_2_eqbump, prices_2)]
eq_delta_3 = [(shift-base)/0.01 for shift, base in zip(prices_3_eqbump, prices_3)]
eq_delta_4 = [(shift-base)/0.01 for shift, base in zip(prices_4_eqbump, prices_4)]

div_delta_1 = [(shift-base)/0.0001 for shift, base in zip(prices_1_divbump, prices_1)]
div_delta_2 = [(shift-base)/0.0001 for shift, base in zip(prices_2_divbump, prices_2)]
div_delta_3 = [(shift-base)/0.0001 for shift, base in zip(prices_3_divbump, prices_3)]
div_delta_4 = [(shift-base)/0.0001 for shift, base in zip(prices_4_divbump, prices_4)]

vols1 = [vols[0]+0.0001, vols[1], vols[2], vols[3]]
vols2 = [vols[0], vols[1]+0.0001, vols[2], vols[3]]
vols3 = [vols[0], vols[1], vols[2]+0.0001, vols[3]]
vols4 = [vols[0], vols[1], vols[2], vols[3]+0.0001]

prices_4_eqvol1 = pricer(S0, q0, r, a, b, vols1, volq, rho, M, h3, h4, mat_position)
prices_4_eqvol2 = pricer(S0, q0, r, a, b, vols2, volq, rho, M, h3, h4, mat_position)
prices_4_eqvol3 = pricer(S0, q0, r, a, b, vols3, volq, rho, M, h3, h4, mat_position)
prices_4_eqvol4 = pricer(S0, q0, r, a, b, vols4, volq, rho, M, h3, h4, mat_position)

eq_vega_1 = [(shift-base)/0.0001 for shift, base in zip(prices_4_eqvol1, prices_4)]
eq_vega_2 = [(shift-base)/0.0001 for shift, base in zip(prices_4_eqvol2, prices_4)]
eq_vega_3 = [(shift-base)/0.0001 for shift, base in zip(prices_4_eqvol3, prices_4)]
eq_vega_4 = [(shift-base)/0.0001 for shift, base in zip(prices_4_eqvol4, prices_4)]

volq1 = [volq[0]+0.0001, volq[1], volq[2], volq[3]]
volq2 = [volq[0], volq[1]+0.0001, volq[2], volq[3]]
volq3 = [volq[0], volq[1], volq[2]+0.0001, volq[3]]
volq4 = [volq[0], volq[1], volq[2], volq[3]+0.0001]

prices_4_divvol1 = pricer(S0, q0, r, a, b, vols, volq1, rho, M, h3, h4, mat_position)
prices_4_divvol2 = pricer(S0, q0, r, a, b, vols, volq2, rho, M, h3, h4, mat_position)
prices_4_divvol3 = pricer(S0, q0, r, a, b, vols, volq3, rho, M, h3, h4, mat_position)
prices_4_divvol4 = pricer(S0, q0, r, a, b, vols, volq4, rho, M, h3, h4, mat_position)

div_vega_1 = [(shift-base)/0.0001 for shift, base in zip(prices_4_divvol1, prices_4)]
div_vega_2 = [(shift-base)/0.0001 for shift, base in zip(prices_4_divvol2, prices_4)]
div_vega_3 = [(shift-base)/0.0001 for shift, base in zip(prices_4_divvol3, prices_4)]
div_vega_4 = [(shift-base)/0.0001 for shift, base in zip(prices_4_divvol4, prices_4)]