import numpy as np
import matplotlib.pyplot as plt

def modified_euler_method(rhs_func, initial_condition, t0, T, h0, N_x, eps):
    t = t0
    h = h0
    v = np.array(initial_condition)
    kounter = [0]
    results = []

    print("{:12.6f} {:12.6f} {:12s} {:12d} {:12.6f} {:12.6f} {:12.6f}".format(
        t, h, "0", kounter[0], *v))

    def euler_Modf(t,v,h):
        v_hat = rhs_func(t, v, kounter)
        v_tilde = rhs_func(t + h, v + h * v_hat, kounter)
        return  v + (h / 2) * (v_hat + v_tilde)



    while t < T and kounter[0] < N_x:
        v_First = euler_Modf(t,v,h)
        v_Second = euler_Modf(t,v,h/2)
        v_Second = euler_Modf(t + h/2, v_Second, h/2)

        R = np.linalg.norm(v_First - v_Second) / (pow(2,2) - 1)

        if R > eps:
            h /= 2

        elif R < (eps / 64):
            h *= 2

        else:
            v = v_First
            t += h

            print("{:12.6f} {:12.6f} {:12.5e} {:12d} {:12.6f} {:12.6f} {:12.6f}".format(
                t, h, R, kounter[0], *v))

        if t + h > T:
            h = T - t

        results.append((t, h, R, kounter[0], *v))

    return results

t0 = 1.5
T = 2.5
h0 = 0.1
N_x = 10000
eps = 0.0001

eps_count = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

kounter_mas = []
def fs(t, v, kounter):
    A = np.array([[-0.4, 0.02, 0], [0, 0.8, -0.1], [0.003, 0, 1]])
    kounter[0] += 1
    return np.dot(A, v)

initial_condition = [1, 1, 2]

result_list = []
for eps in eps_count:
    print("Eps = ", eps)
    results = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    result_list.append(results)
    print(" ")

for i, results in enumerate(result_list):
    t_values = [r[0] for r in results]
    h_values = [r[1] for r in results]
    plt.plot(t_values, h_values, label = f'eps={eps_count[i]}')
plt.xlabel('t')
plt.ylabel('h')
plt.title('Изменение шага по отрезку для разных точностей')
plt.legend()
plt.show()

min_h_values = [min([r[1] for r in results]) for results in result_list]
plt.plot(eps_count, min_h_values)
plt.xlabel('eps')
plt.ylabel('мигимальное h')
plt.title('Зависимость минимального шага от заданной точности')
plt.show()

num_steps = [len(results) for results in result_list]
plt.loglog(eps_count, num_steps)
plt.xlabel('eps')
plt.ylabel('число шагов')
plt.title('Зависимость числа шагов от заданной точности')
plt.show()

for i, results in enumerate(result_list):
    t_values = [r[0] for r in results]
    v_values = [r[3:] for r in results]
    plt.loglog(t_values, v_values, label = f'eps = {eps_count[i]}')
plt.xlabel('t')
plt.ylabel('v')
plt.title('Решение для разных значений заданной точности')
plt.legend()
plt.show()

