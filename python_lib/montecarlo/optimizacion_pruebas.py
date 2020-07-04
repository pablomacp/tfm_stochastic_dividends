# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:37:59 2020

@author: pablo
"""

from montecarlo import *
from funciones import cal_yf_from_mat, days_yf, parametros_to_pasos
from math import exp
from scipy.optimize import fsolve, root

def pricer(S0, q0, r, a, b, vols, volq, rho, M, h_elegido, hstart):
    a_pasos = parametros_to_pasos(a, t0, T_futdiv)
    volq_pasos = parametros_to_pasos(volq, t0, T_futdiv)
    vols_pasos = parametros_to_pasos(vols, t0, T_opt_sx5e)

    # Número de pasos N en la simulación
    N = len(h_elegido)

    # Cálculos usando normrnd y correlacionando variables después
    S, q = HybridStockDividendsMSamples(S0, q0, r, a_pasos, b, vols_pasos, volq_pasos, rho, M, N, h_elegido)

    payoffs_divfut = PayoffDivFut(S[1:N + 1], q[1:N + 1], h_elegido, hstart)
    payoffs_divopt = exp(-r * yf_futdiv[0]) * PayoffOptCall(payoffs_divfut, K_div)
    payoffs_eqopt = exp(-r * yf_futdiv[0]) * PayoffOptCall(S[N], S0)

    return [np.mean(payoffs_divfut), np.mean(payoffs_divopt), np.mean(payoffs_eqopt)]

def f_objetivo(S0, q0, r, a, b, vols, volq, rho, M, h_elegido, mat_position, objetivos):
    prices = pricer(S0, q0, r, a, b, vols, volq, rho, M, h_elegido, mat_position)
    return [x1 - x2 for x1, x2 in zip(prices, objetivos)]

if __name__ == '__main__':
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
    hstart = [0] + [len(h[i]) for i in range(len(h)-1)]

    # h1 = h[0]
    # h2 = h[1]
    # h3 = h[2]
    # h4 = h[3]

    div_fut_prices = [53.1, 49.8, 63.6, 67.4]
    div_call_opt_prices = [5.60, 4.19, 11.02, 16.13]  # Excel. Tomo el strike 65, que es el más líquido.
    eurostoxx_call_opt_prices = [239.0016, 292.9, 312.5604, 336.7692]

    # Datos del problema
    S0 = 2680.3  # También es el strike de las call sobre el eurostoxx
    q0 = 0.022794603
    M = 2 ** 14
    K_div = 65  # Strike dividend call options
    rho = 0.23240868 # -0.088195234
    r = -0.00168

    # Parametros a calibrar
    b = 0.001  # Criterio experto
    a = [0.01, 0.01, 0.01, 0.01]
    vols = [0.1, 0.1, 0.1, 0.1]
    volq = [0.1, 0.1, 0.1, 0.1]

    # Objetivos
    objetivos = [[div_fut_prices[i], div_call_opt_prices[i], eurostoxx_call_opt_prices[i]] for i in range(4)]

    prices_0 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h[0], hstart[0])

    initial_guess = [0.2, 0.5, 0.5]
    metodo = 'broyden1'
    f0 = lambda x: f_objetivo(S0,
                              q0,
                              r,
                              [x[0], a[1], a[2], a[3]],
                              b,
                              [x[1], vols[1], vols[2], vols[3]],
                              [x[2], volq[1], volq[2], volq[3]],
                              rho,
                              M,
                              h[0],
                              hstart[0],
                              objetivos[0])
    # [a_sol0, vols_sol0, volq_sol0] = fsolve(f0, initial_guess)
    sol0 = root(f0, initial_guess, method=metodo, tol=10**-4)
    a[0] = sol0['x'][0]
    vols[0] = sol0['x'][1]
    volq[0] = sol0['x'][2]
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # print(f_objetivo(S0,
    #                           q0,
    #                           r,
    #                           [a[0], a[0], a[2], a[3]],
    #                           b,
    #                           [vols[0], vols[0], vols[2], vols[3]],
    #                           [volq[0], volq[0], volq[2], volq[3]],
    #                           rho,
    #                           M,
    #                           h[1],
    #                           hstart[1],
    #                           objetivos[1]))
    # a = f_objetivo(S0,               q0,               r,               [a[0], a[0], a[2], a[3]],               b,               [vols[0], vols[0], vols[2], vols[3]],               [volq[0], volq[0], volq[2], volq[3]],               rho,               M,               h[1],               hstart[1],               objetivos[1])
    f1 = lambda x: f_objetivo(S0,
                              q0,
                              r,
                              [a[0], x[0], a[2], a[3]],
                              b,
                              [vols[0], x[1], vols[2], vols[3]],
                              [volq[0], x[2], volq[2], volq[3]],
                              rho,
                              M,
                              h[1],
                              hstart[1],
                              objetivos[1])
    # [a_sol1, vols_sol1, volq_sol1] = fsolve(f1, initial_guess)
    sol1 = root(f1, initial_guess, method=metodo, tol=10**-6)
    a[1] = sol1['x'][0]
    vols[1] = sol1['x'][1]
    volq[1] = sol1['x'][2]
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    f2 = lambda x: f_objetivo(S0,
                              q0,
                              r,
                              [a[0], a[1], x[0], a[3]],
                              b,
                              [vols[0], vols[1], x[1], vols[3]],
                              [volq[0], volq[1], x[2], volq[3]],
                              rho,
                              M,
                              h[2],
                              hstart[2],
                              objetivos[2])
    # [a_sol2, vols_sol2, volq_sol2] = fsolve(f2, initial_guess)
    sol2 = root(f2, initial_guess, method=metodo, tol=10**-6)
    a[2] = sol2['x'][0]
    vols[2] = sol2['x'][1]
    volq[2] = sol2['x'][2]
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    f3 = lambda x: f_objetivo(S0,
                              q0,
                              r,
                              [a[0], a[1], a[2], x[0]],
                              b,
                              [vols[0], vols[1], vols[2], x[1]],
                              [volq[0], volq[1], volq[2], x[2]],
                              rho,
                              M,
                              h[3],
                              hstart[3],
                              objetivos[3])
    # [a_sol3, vols_sol3, volq_sol3] = fsolve(f3, initial_guess)
    sol3 = root(f3, initial_guess, method=metodo, tol=10**-6)
    a[3] = sol3['x'][0]
    vols[3] = sol3['x'][1]
    volq[3] = sol3['x'][2]
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    print('Param  solver 1:', sol0['x']) # 2^15 0.015506635921427488 0.29514724550393595 0.1656479699860158
    # 2^16 0.015512465425311258 0.29648353534605315 0.16555735571639657
    # 2^17 0.01549042738583047 0.29837489136838047 0.16497912181648908
    # 2^18 0.015541662125404594 0.2970416411412151 0.16578548959864273
    # 2^19 0.015539355389007983 0.29778113130425526 0.1658044809216574

    print('Param  solver 2:', sol1['x'])
    print('Param  solver 3:', sol2['x'])
    print('Param  solver 4:', sol3['x'])

    prices_1 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h[0], hstart[0])
    prices_2 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h[1], hstart[1])
    prices_3 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h[2], hstart[2])
    prices_4 = pricer(S0, q0, r, a, b, vols, volq, rho, M, h[3], hstart[3])

    # print('Precios sin calibración:', prices_0)
    print('Precios con calibración 1:', prices_1)
    print('Precios de mercado 1', objetivos[0], '\n')
    print('Precios con calibración 2:', prices_2)
    print('Precios de mercado 2', objetivos[1], '\n')
    print('Precios con calibración 3:', prices_3)
    print('Precios de mercado 3', objetivos[2], '\n')
    print('Precios con calibración 4:', prices_4)
    print('Precios de mercado 4', objetivos[3], '\n')

    ########################
    ######################## Está mal calculado el índice de dividendos, hay que resetearlo cada año