---
layout: post
title: "La magia de la vectorización"
description: >
  Qué es la vectorización en Python con la librería NumPy para acelerar tu código. Guía clara con ejemplos basados en algoritmos genéticos.

tags:
  - algoritmos genéticos
  - optimización
  - vectorización
  - machine learning
  - python
  - numpy
  - eficiencia
---

## ¿Qué es la vectorización?

Como programador siempre he tenido en cuenta la eficiencia a la hora de crear sistemas. Los numeritos indicando poco tiempo de ejecución nos producen cosquillas en el cerebro. Una forma de conseguir código más veloz es con la técnica de vectorización.

¿Qué es la vectorización? Es un método por el cual aplicamos operaciones sobre un conjunto de elementos de forma simultánea, en vez de aplicarlas de una en una. Sencillo de entender y muy fácil de aplicar en *Python* con la librería de cálculo numérico *NumPy*.

{% include image.html
   path="/assets/images/vectorizacion-en-python/vectorization.png"
   caption="Ejemplo visual de operación vectorizada sobre un conjunto de elementos"
   width="500"
%}

Sin embargo, no todos los problemas son vectorizables. Para que una operación pueda vectorizarse, los cálculos sobre cada elemento deben ser independientes entre sí, es decir, el resultado de un elemento no puede depender del resultado de otro. Esto es así porque las operaciones se ejecutan en paralelo. Si existiera dependencia entre los resultados, sería necesario coordinar la ejecución, introduciendo un overhead que reduciría, o incluso anularía, las ventajas en rendimiento.

También es necesario estructurar los datos en *arrays* o matrices, es decir, estructuras homogéneas. Las *GPUs* y *CPUs* están optimizadas para operar bajo estas estructuras. *NumPy* esta programado para aprovechar estas optimizaciones.

## Vectorización de un GA

Como ya vimos en mi anterior *post* cómo funcionan los [algoritmos genéticos](/2025/12/30/algoritmos-geneticos/#vectorizacion-de-un-ga), podemos aprovechar el código secuencial en *Python* puro para introducir unas cuantas mejoras. Todos los operadores de ese código son perfectamente vectorizables. Al realizar el cambio, veremos una mejora muy sustancial, no solo por el uso de esta técnica, sino porque los bucles en *Python* son inherentemente lentos. 

Empecemos por el operador de mutación:

```python
def mutate(self, x: np.ndarray) -> np.ndarray:
        for i in range(x.size):
            if np.random.rand() < self.mutation_rate:
                x[i] = 1 - x[i]
        return x
```

```python
def mutate_vectorized(self, population: np.ndarray) -> np.ndarray:
        mask = (
            np.random.rand(population.shape[0], population.shape[1])
            < self.mutation_rate
        )
        population[mask] = 1 - population[mask]
        return population
```

En el primer código usé un `for-loop` en *Python* (extremadamente lento si el conjunto a iterar crece mucho) para ir mutando cada gen uno a uno. Además se realiza la operación de generación de un número aleatorio individualmente.

En el segundo código creamos una máscara del tamaño de la población total. Véase que aquí operamos no sobre un individuo $x\in X$, sino sobre toda la población $X$. Si el número generado es menor a la probabilidad de mutación, entonces se aplica mutación. Después se aplica, de forma simultánea, la máscara sobre la operación para hacer el cambio de *bit*.

La diferencia es abismal. Asumamos unos $800$ genes por cromosoma con $50$ cromosomas y $200$ generaciones. En el primer código estaríamos iterando unas $800$ veces por cada individuo (unas $50\times 800=40.000$ iteraciones de un bucle *for*). Eso multiplicado por $200$ generaciones da unas $8.000.000$ iteraciones totales. En el segundo código se muta a la población de una generación de una sola vez. Pasamos de $8$ millones de iteraciones a unas $200$ operaciones vectoriales masivas.
