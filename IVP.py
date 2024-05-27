import numpy as np
import matplotlib.pyplot as plt

def modified_euler_method(rhs_func, initial_condition, t0, T, h0, N_x, eps):
    t = t0
    h = h0
    v = np.array(initial_condition)
    kounter = [0]
    results = []
    steps = []
    solutions = []
    coord = []

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
            steps.append(h)
            solutions.append(v.copy())
            coord.append(t)

            print("{:12.6f} {:12.6f} {:12.5e} {:12d} {:12.6f} {:12.6f} {:12.6f}".format(
                t, h, R, kounter[0], *v))

        if t + h > T:
            h = T - t

        results.append((t, h, R, kounter[0], *v))

    return results, steps, solutions, coord

t0 = 1.5
T = 2.5
h0 = 0.1
N_x = 10000
eps = 0.0001
n = 3
eps_count = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

kounter_mas = []

function_code = []
for i in range(n+3):
    line = input()
    function_code.append(line)

func = '\n'.join(function_code)
exec(func)

#def fs(t, v, kounter):
#    A = np.array([[-0.4, 0.02, 0], [0, 0.8, -0.1], [0.003, 0, 1]])
#    kounter[0] += 1
#    return np.dot(A, v)

initial_condition = [1, 1, 2]

results, steps, solutions, coord = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)

result_list = []
for eps in eps_count:
    print("Eps = ", eps)
    results = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    result_list.append(results)
    print(" ")


fig, axes = plt.subplots(len(eps_count), 1, figsize=(8, 8))
for i, eps in enumerate(eps_count):
    results, steps, solutions, coord = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    axes[i].plot(coord, steps)
    axes[i].set_xlabel("Координата t")
    axes[i].set_ylabel("Величина шага h")
    axes[i].set_title(f"Изменение шага, eps={eps}")

plt.tight_layout()
plt.show()

min_steps = []

for i, eps in enumerate(eps_count):
    results, steps, solutions, coord = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    min_steps.append(min(steps))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(eps_count, min_steps)
ax.set_xlabel('Точность (eps)')
ax.set_ylabel('Минимальный шаг')
ax.set_title('Зависимость минимального шага от точности')
plt.show()

num_steps = []

for i, eps in enumerate(eps_count):
    results, steps, solutions, coord = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    num_steps.append(len(steps))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(eps_count, num_steps)
ax.set_xlabel('Точность (eps)')
ax.set_ylabel('Количество шагов')
ax.set_title('Зависимость количества шагов от точности')
plt.show()

fig, axes = plt.subplots(len(eps_count), 1, figsize=(8, 8))
for i, eps in enumerate(eps_count):
    results, steps, solutions, coord = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    solutions = np.array(solutions)
    for j in range(solutions.shape[1]):
        axes[i].plot(coord, solutions[:, j], label=f'v{j+1}')
    axes[i].set_xlabel("Координата t")
    axes[i].set_ylabel("Значение решения")
    axes[i].set_title(f"Изменение решения, eps={eps}")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()



