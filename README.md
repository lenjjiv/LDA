## Linear Discriminant Analysis

**Linear Discriminant Analysis** (LDA, или линейный дискриминант Фишера) — алгоритм понижения размерности, который, как ни странно, является supervised-алгоритмом. 

Он стремится **максимиpовать межклассовую** дисперсию и **минимизировать внутриклассовую**, используя предложенные метки, а затем делает ортогональную проекцию на признаковое подпространство.

### **Формулы**
1. **SW (внутриклассовая корреляционная матрица)**:
   
   $$S_W = \sum_{c} \sum_{x \in c} (x - \mu_c)(x - \mu_c)^T$$
   
   Где:
   - $\mu_c$ — среднее значение в классе $c$.
   - $x$ — точка данных из класса $c$.

> (здесь мы суммируем отклонения всех точек от центров своих классов)

2. **SB (межклассовая корреляционная матрица)**:
   
   $$S_B = \sum_{c} N_c (\mu_c - \mu)(\mu_c - \mu)^T$$
   
   Где:
   - $\mu$ — общее среднее.
   - $N_c$ — количество точек в классе $c$.

> (здесь мы как бы "концентрируем" всю "массу" класса в его центре и ищем суммарное отклонение их как единого целого от общего центра)

3. **Диагонализация (приведение к собственным осям)**:
   
   $$S_W^{-1} S_B v = \lambda v$$
   
   Где:
   - $v$ — собственные векторы (основные направления).
   - $\lambda$ — собственные значения (вес каждой оси).

Векторы, соответствующие наибольшим собственным значениям максимизируют межклассовую и минимизируют внутриклассовую дисперсию в данных.

Далее сортируем полученные оси в порядке убывания собственных чисел и используем "максимально информативные" оси (в количестве, в котором нам это нужно), — понижая размерность за счёт наименьших собственных чисел.