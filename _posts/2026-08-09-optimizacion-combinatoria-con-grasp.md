---
layout: post
title: "Optimización combinatoria con GRASP"
description: >
  La optimización es un tema que me apasiona desde hace años. El primer post que subí a este blog ya mencionaba las metaheurísticas y en el segundo traté los algoritmos genéticos. En este post explicaré el algoritmo GRASP, cómo funciona y por qué funciona. Es un algoritmo de optimización pseudo-estocástico de diseño simple, pero con grandes aplicaciones.

tags:
  - ia
  - optimización
  - inteligencia artificial
  - optimization
  - grasp
  - greedy
  - metaheuristics
  - tsp
  - artificial intelligence
---

## No todo se resuelve con un algoritmo exacto
No todos los problemas son igual de complicados. Cuando hablo de la complejidad de un problema, a lo que me refiero es a los recursos que consume una serie de pasos definidos para poder hallar una solución a ese problema. Por lo general, los recursos que utiliza un algoritmo (la serie de pasos definidos y ordenados para solucionar un problema) suelen ser dos, tiempo y espacio. El espacio es la cantidad de información que debemos guardar para poder resolver ese problema. Por ejemplo, ¿cuál es la complejidad de la codificación numérica? Si quiero escribir un número en decimal, por ejemplo $1365$, necesito $4$ caracteres (dígitos). En cambio, para $13454390$ necesito solo $8$ caracteres, pese a que el valor haya crecido enormemente, la cantidad de información para representarlo no lo ha hecho tanto. En general, ya sea en binario, decimal o con otra codificación, si el número es $x$, el número de caracteres que necesito es de orden $n = log(x)$. Esa es su complejidad en espacio definida como una función.

La complejidad temporal indica cuánto trabajo computacional necesita un algoritmo para resolver un problema. Tanto la complejidad en espacio como la temporal dependen del tamaño de entrada del problema. En el ejemplo anterior el tamaño de entrada es el valor del número en sí.


{% include image.html
   path="/assets/images/optimizacion-combinatoria-con-grasp/time-complexity.png"
   caption="Tabla de tasas de crecimiento comparadas."
   width="1000"
%}

En la tabla se muestran distintos tipos de crecimientos con respecto al tamaño de la entrada. Un problema con complejidad exponencial puede resolverse de forma muy rápida para entradas pequeñas, pero en cuanto crece un poco el tiempo de resolución sale disparado. Por ello, conocer la complejidad de un problema de antemano es necesario en muchos casos para saber aterrizar las expectativas.

El hecho de que existan algoritmos exactos para algunos de estos complejísimos problemas ya es algo por lo que dar gracias. Otros problemas no tienen solución conocida (y posiblemente sea necesario aplicar algoritmos de aprendizaje para encontrar soluciones suficientemente buenas, como bien expliqué en [mi primer post](/2025/12/25/de-que-trata-este-blog)). Pero claro, ¿de qué sirve un programa que resuelve el problema si te dará la respuesta en unos pocos años? No es viable esperar esa cantidad de tiempo. Por ello es que existen métodos mucho más rápidos, pero que sacrifican la exactitud por la velocidad. Dentro de estos métodos están, por ejemplo, los [algoritmos genéticos](/2025/12/30/algoritmos-geneticos) de los que ya hablé.

Hoy, explicaré otro de estos métodos de búsqueda. El conocido como **GRASP** (*Greedy Randomized Adaptive Search Procedure*).

## Problemas combinatorios
El algoritmo de *GRASP* se utiliza sobre todo en resolución de problemas combinatorios. Un problema es combinatorio cuando las soluciones se construyen mediante combinaciones de elementos discretos, como por ejemplo [el problema de la mochila](/2025/12/30/algoritmos-geneticos.html#el-problema-de-la-mochila), en el que se debe seleccionar un conjunto de objetos a guardar en una mochila, maximizando el valor total y sin pasarse del peso máximo permitido. Este tipo de problemas se pueden solucionar a lo bruto, que es lo primero que se suele intentar. En este caso, procedo a crear todas las combinaciones de objetos dada la restricción del peso máximo. Después, simplemente cojo aquella que tenga mayor valor. ¿Sencillo? Sí. ¿Eficiente? Como tengas una mochila de muchos litros quizá te pases eones esperando a la solución final...

La complejidad del problema de la mochila, resuelto por fuerza bruta, es de $O(2^n)$. Sin entrar en detalles, la $O$ de $O(n)$ viene de *"Big O"* y sirve para describir cómo crece el coste de un algoritmo cuando aumenta el tamaño de la entrada. En este caso significa que, por fuerza bruta, el número de pasos/operaciones crece con la función $2^n$. Puesto que cada objeto tiene dos posibilidades (cogerlo o no cogerlo) y existen $n$ objetos totales, en el peor de los casos habremos realizado un total de $2^n$ operaciones.

### El problema del viajante (TSP)
El problema del viajante o *Traveling Salesman Problem* en inglés es el ["hola mundo"](https://akkologlu.medium.com/hello-world-the-story-of-the-legendary-first-step-in-programming-bb06a623f7f7) de la optimización. Dada una lista de ciudades y las distancias entre cada par de ellas, ¿cuál es la ruta más corta posible que visite cada ciudad exactamente una vez y al final regrese a la ciudad de origen? El **TSP**, por sus siglas en inglés, tiene diversas aplicaciones en planificación y logística. Es por ello que voy a codificar en *Python* cómo funciona y voy a resolverlo tanto por fuerza bruta como con *GRASP*.

### Definición formal
El **TSP** puede ser formulado de la siguiente forma:
- Sea $x_{ij}$ igual a $1$, si se elige un camino entre la ciudad $i$ y la ciudad $j$, $0$ en otro caso, para el conjunto de ciudades $0,...,n$. Sea $d_{ij}$ la distancia entre la ciudad $i$ y la ciudad $j$. El conjunto de todas las ciudades es $V$. Dadas estas variables se define el problema de optimización con la función objetivo:

$$min\sum_{i=0}^n\sum_{j=0}^nd_{ij}x_{ij}$$

- Es decir, se intenta minimizar la distancia total viajada. La restricción es que cada ciudad solo se debe visitar una vez. Es decir, de cada ciudad se sale una vez:

$$\sum_{j=0}^nx_{ij}=1, \forall i\in V$$

- A cada ciudad se entra solo una vez:

$$\sum_{i=0}^nx_{ij}=1, \forall j\in V$$

- Y no se puede entrar a la misma ciudad de la que se salió:

$$x_{ii}=0, \forall i\in V$$

Este problema es mucho más complejo que el de la mochila. Donde antes se buscaban todas las combinaciones sin importar el orden, ahora se buscan todas las **permutaciones**. Ya no es lo mismo el camino $A\rightarrow B$ que el camino $B\rightarrow A$. Esa pequeña diferencia hace que el problema ya no sea exponencial, sino factorial: $O(n!)$. En términos de crecimiento es una pesadilla.

## TSP en Python
Para poder resolver el problema, primero debemos representarlo. Lo que he hecho es representar en un plano $2D$ las coordenadas de cada ciudad. De esta forma es muy sencillo calcular posteriormente la distancia entre ellas y además podemos visualizar el grafo, que siempre queda muy bonito y hace que cale todo mucho mejor. La siguiente función crea una matriz $N\times 2$ de forma aleatoria donde $N$ es el número de ciudades. Cada ciudad tendrá dos valores, las coordenadas.

```python
def create_cities(num_cities: int) -> np.ndarray:
    """Create random coordinates representing a set of cities.

    Args:
        num_cities (int): Number of cities.

    Returns:
        np.ndarray: A matrix containing the coordinates of each city.
    """

    return np.random.rand(num_cities, 2)
```

Ahora hay que calcular la matriz de distancias. En esta matriz, cada elemento de una fila corresponde a una ciudad y cada elemento de una columna repite las ciudades. De esta forma, codificamos en una estructura matemática las distancias entre cada par de ciudades. Obviamente, esta matriz contendrá la distancia entre una ciudad y ella misma. Estas posiciones corresponden a la diagonal de la matriz. Una forma de hacer este cálculo sería aplicar dos bucles sobre la matriz, de forma que fuese uno a uno computando las distancias. Por suerte, en *Numpy* existe lo que se conoce como ["broadcasting"](https://numpy.org/doc/stable/user/basics.broadcasting.html), lo cual permite hacer esta operación mucho más eficiente (todos los cálculos se hacen a la vez). En otro post, explicaré en detalle cómo funciona. 

La distancia es la distancia euclídea, que en dos dimensiones es:

$$dist = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$$

En *Numpy* tenemos la conveniente función de `np.linalg.norm`.

```python
def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Create a symmetric distance matrix from a set of city coordinates.

    Args:
        cities (np.ndarray): Matrix containing the coordinates of each city.

    Returns:
        np.ndarray: A symmetric matrix containing the distances between cities.
    """

    diff = cities[:, None, :] - cities[None, :, :]
    return np.linalg.norm(diff, axis=-1)
```

Con estas dos funciones ya podemos visualizar las distintas ciudades en un gráfico y aplicar distintos algoritmos de resolución para encontrar el camino más eficiente de reparto.

{% include image.html
   path="/assets/images/optimizacion-combinatoria-con-grasp/cities-example.png"
   caption="Grafo de ciudades."
   width="400"
%}

## Fuerza bruta

{% include image.html
   path="/assets/images/optimizacion-combinatoria-con-grasp/tsp-brute-force.gif"
   caption="Resolución de una instancia de TSP con fuerza bruta. Fuente: https://es.wikipedia.org/wiki/Problema_del_viajante."
   width="1000"
%}

La solución de **TSP** por fuerza bruta no tiene misterio. Debemos crear todas las permutaciones (no combinaciones, aquí sí importa el orden) de caminos posibles dadas las restricciones de que solo se puede pasar una vez por cada ciudad, hay que pasar por todas y hay que volver a la ciudad de origen. Después, seleccionar aquella que tenga la menor distancia total recorrida.

De nuevo, en *Python* existe la librería `itertools` con funciones muy convenientes como `permutations`, que nos permite iterar sobre todas las permutaciones posibles de un conjunto dado.

Las siguientes funciones calculan el coste de un camino (necesitamos la matriz de distancia para ver cuánta distancia hay entre dos ciudades y la lista de ciudades que compone un camino) y el algoritmo de búsqueda del camino más corto por fuerza bruta. Es muy sencillo, vamos iterando por cada permutación y la vamos evaluando. Si es la mejor hasta ahora, la guardamos. Así hasta expresar todas las combinaciones. Existen maneras mucho más eficientes de hacer la búsqueda y asegurar que sea exacta, sin metaheurísticas y con garantías, pero hace falta aplicar algoritmos de programación dinámica y estoy bastante oxidado (hay que practicar más *Leetcode*).

```python
def tsp_cost(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Compute the total cost of a TSP tour.

    The tour is treated as a closed cycle, including the return from the
    last city to the starting city.

    Args:
        tour (np.ndarray): Ordered array of city indices.
        distance_matrix (np.ndarray): Pairwise distance matrix.

    Returns:
        float: Total cost of the tour.
    """

    cost = 0.0

    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]

    cost += distance_matrix[tour[-1], tour[0]]

    return cost


def solve_tsp_brute_force(distance_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the Traveling Salesman Problem (TSP)

    Args:
        distance_matrix (np.ndarray): A square symmetric matrix of shape
            (n, n), where distance_matrix[i, j] represents the distance
            between cities i and j.

    Returns:
        tuple[np.ndarray, float]:
            - The optimal tour as a NumPy array of city indices, starting
              at city 0.
            - The total length (cost) of the optimal tour, including the
              return from the last city back to the starting city.
    """
    n = distance_matrix.shape[1]
    best_tour = None
    best_cost = np.inf

    for rest in permutations(range(1, n)):
        tour = (0,) + rest
        tour_array = np.array(tour)
        tour_cost = tsp_cost(tour_array, distance_matrix)

        if best_cost > tour_cost:
            best_cost = tour_cost
            best_tour = tour_array

    return best_tour, best_cost
```
## GRASP
El algoritmo *GRASP* viene a ser una mejora del famoso algoritmo *greedy* o voraz, por tanto convendría programar una alternativa puramente *greedy* para compararla tanto con el método por fuerza bruta como con *GRASP*. 

El método *greedy*, si bien es rápido y obtiene buenas soluciones, muchas veces no es suficiente. Estos tipos de búsqueda tienen una limitación importante, y es que son incapaces de explorar el espacio de búsqueda pues su naturaleza se basa en moverse al siguiente mejor estado, sin considerar nada más que el estado actual. Si el mejor movimiento es moverse de la ciudad $A$ a la ciudad $B$, lo hace. Pero puede ocurrir que, lo que a priori es un mal movimiento, termine resultando en su conjunto una mejor solución. Por ello muchas metaheurísticas incluyen procesos estocásticos, a modo de exploración. En el caso de *GRASP* se abordan dos fases generales:

### Exploración
Empezando por la ciudad de origen, *GRASP* construye de forma incremental una solución seleccionando la siguiente ciudad desde una lista de candidatos restringida (**RCL**). Esta lista contiene a los mejores candidatos en función del criterio *greedy* (es decir, para *TSP* es la menor distancia a la ciudad actual). Desde esta lista, para romper con el determinismo de *greedy*, se elige el candidato al azar. Esta es la fase "exploratoria" y puede controlarse mediante el parámetro $\alpha$.

Este parámetro determina hasta qué punto se relaja el criterio *greedy* para construir la *RCL*. Para ello, se establece un umbral entre el coste del mejor y el peor candidato:

$$
\text{threshold} = C_{\min} + \alpha(C_{\max} - C_{\min})
$$

donde $C_{\min}$ y $C_{\max}$ representan, respectivamente, la menor y la mayor distancia entre los candidatos disponibles. De esta forma, $\alpha$ determina qué proporción del intervalo entre ambos extremos se considera aceptable.

Cuando $\alpha=0$, el umbral coincide con $C_{\min}$, por lo que únicamente el mejor candidato forma parte de la *RCL*. En consecuencia, el comportamiento es equivalente al de un algoritmo *greedy*. A medida que aumenta $\alpha$, el umbral se desplaza hacia $C_{\max}$ y se incorporan más candidatos a la *RCL*, aumentando la diversidad de las soluciones construidas. Con $\alpha=1$, todos los candidatos son aceptados y la siguiente ciudad se selecciona completamente al azar.

### Explotación
Una vez se construye una solución siguiendo un método *pseudo-greedy*, se explota mediante una búsqueda local. Este tipo de búsqueda se basa en cambios menores sobre la solución original, cambios en su vecindario más cercano. Son métodos muy rápidos y sencillos de implementar que intentan mejorar al máximo la solución actual, por ello son algoritmos de fase de explotación. Existen muchas implementaciones, pero en mi caso he implementado una búsqueda local sencilla, basada en intercambiar pares de ciudades de posición y aceptar la primera que mejore el costo. La búsqueda termina cuando ya no mejora más.

La fase de exploración y explotación se repiten durante una serie de iteraciones definidas, hasta que no haga mejoras significativas o incluso hasta un tiempo máximo definido. El criterio de parada es muy variado. El código en *Python* es el siguiente:

```python
class TSP_GRASP:
    def __init__(self, max_iterations: int = 20, alpha: float = 0.2):
        self.max_iterations = max_iterations
        self.alpha = alpha

    def build_solution(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Construct a feasible TSP solution using the GRASP construction phase.

        Starting from city 0, the solution is built incrementally by selecting
        the next city from a Restricted Candidate List (RCL). The RCL contains
        the best candidates according to the greedy criterion, and one of them
        is chosen uniformly at random.

        Args:
            distance_matrix (np.ndarray): Pairwise distance matrix.

        Returns:
            np.ndarray: A feasible TSP tour represented as an ordered array of
            city indices.
        """

        n = distance_matrix.shape[1]

        unvisited = np.array([i for i in range(1, n)])
        candidates = []
        solution = np.zeros(1, dtype=np.int64)

        for _ in range(n - 1):
            for city in unvisited:
                cost = distance_matrix[solution[-1], city]

                candidates.append((city, cost))

            costs = [cost for _, cost in candidates]
            min_cost = min(costs)
            max_cost = max(costs)

            threshold = min_cost + self.alpha * (max_cost - min_cost)
            rcl = [city for city, cost in candidates if cost <= threshold]
            next_city = np.random.choice(rcl)
            solution = np.append(solution, next_city)

            unvisited = np.setdiff1d(unvisited, [next_city])
            candidates = []
        return np.array(solution)

    def local_search(
        self, solution: np.ndarray, distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Improve a TSP solution using a swap-based local search.

        The algorithm iteratively swaps pairs of cities and accepts the first
        improving move. The process stops when no improving swap is found.

        Args:
            solution (np.ndarray): Initial TSP tour.
            distance_matrix (np.ndarray): Pairwise distance matrix.

        Returns:
            np.ndarray: Locally improved TSP tour.
        """

        best_solution = solution.copy()
        best_cost = tsp_cost(best_solution, distance_matrix)

        improved = True

        while improved:
            improved = False

            for i in range(1, len(solution) - 1):
                for j in range(i + 1, len(solution)):
                    candidate = best_solution.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]

                    candidate_cost = tsp_cost(candidate, distance_matrix)

                    if candidate_cost < best_cost:
                        best_solution = candidate
                        best_cost = candidate_cost
                        improved = True
                        break

                if improved:
                    break

        return best_solution

    def optimize(self, distance_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        best_solution = None
        best_cost = np.inf

        for _ in range(self.max_iterations):
            solution = self.build_solution(distance_matrix)

            solution = self.local_search(solution, distance_matrix)

            cost = tsp_cost(solution, distance_matrix)

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        return best_solution, best_cost
```

## Resultados
Ha llegado la hora de comparar las tres opciones propuestas, la fuerza bruta, el voraz básico y *GRASP*. Aquí los resultados para *TSP* con $5, 9, 11$ y $12$ ciudades:

|  Algorithm  | n_cities | execution_time_s |      tour_cost     |
| :---------: | :------: | :--------------: | :----------------: |
| Brute Force |     5    |     0.000052     | 2.1657752216052906 |
|    Greedy   |     5    |     0.000018     | 2.1821739012227956 |
|    GRASP    |     5    |     0.004553     | 2.1657752216052906 |

{% include image_grid.html 
   paths="/assets/images/optimizacion-combinatoria-con-grasp/cities-5.png|
   /assets/images/optimizacion-combinatoria-con-grasp/best_tour-5.png"
   caption="Grafo de las 5 ciudades frente a la solución encontrada por GRASP."
%}

|  Algorithm  | n_cities | execution_time_s |      tour_cost     |
| :---------: | :------: | :--------------: | :----------------: |
| Brute Force |     9    |     0.107380     | 2.6180824541699557 |
|    Greedy   |     9    |     0.000035     | 3.0401040631190206 |
|    GRASP    |     9    |     0.013204     | 2.6180824541699557 |

{% include image_grid.html 
   paths="/assets/images/optimizacion-combinatoria-con-grasp/cities-9.png|
   /assets/images/optimizacion-combinatoria-con-grasp/best_tour-9.png"
   caption="Grafo de las 9 ciudades frente a la solución encontrada por GRASP."
%}

|  Algorithm  | n_cities | execution_time_s |      tour_cost     |
| :---------: | :------: | :--------------: | :----------------: |
| Brute Force |    11    |     10.690878    | 2.8734761900015102 |
|    Greedy   |    11    |     0.000031     | 3.3140517561088605 |
|    GRASP    |    11    |     0.014780     | 2.8734761900015102 |

{% include image_grid.html 
   paths="/assets/images/optimizacion-combinatoria-con-grasp/cities-11.png|
   /assets/images/optimizacion-combinatoria-con-grasp/best_tour-11.png"
   caption="Grafo de las 11 ciudades frente a la solución encontrada por GRASP."
%}

|  Algorithm  | n_cities | execution_time_s |     tour_cost     |
| :---------: | :------: | :--------------: | :---------------: |
| Brute Force |    12    |    127.815707    | 3.439504983165455 |
|    Greedy   |    12    |     0.000028     | 4.177233605024462 |
|    GRASP    |    12    |     0.019386     | 3.439504983165455 |

{% include image.html
   path="/assets/images/optimizacion-combinatoria-con-grasp/execution_time_vs_n_cities.png"
   caption="Tiempo de ejecución de distintos algoritmos según crece el tamaño de entrada en TSP."
   width="1000"
%}

Como puede observarse, el algoritmo por fuerza bruta no es una opción que pueda utilizarse en la vida real. Ni paralelizándolo ni usando una versión más eficiente por programación dinámica sería usable, todas las versiones alcanzarían en algún punto tiempos imposibles. Mientras tanto, las versiones de *greedy* y *GRASP* pueden crecer y crecer y seguir siendo veloces al encontrar la solución. En estos resultados además se observa la diferencia clave entre ambos. El algoritmo voraz no encuentra nunca el óptimo global en las pruebas realizadas, mientras que *GRASP* lo hace siempre. 

Hay que entender que *GRASP* no garantiza el óptimo global como sí lo hace la fuerza bruta, pero es capaz de aproximar el valor y obtener soluciones suficientemente buenas. Todo ello haciendo un simple cambio al *greedy* de toda la vida.